# Minimal production image: the FastAPI app, its migrations, and the
# evaluation harness -- mirrors this monorepo's own docintel/Dockerfile
# structure (see its own comments for the same reasoning).
#
# Override PYTHON_IMAGE to build from a registry mirror or an internal
# registry, e.g.
#   docker build --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.13-slim .
ARG PYTHON_IMAGE=python:3.13-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
# The evaluation harness and its committed fixtures, so `python -m
# evals.run --suite retrieval` / `--suite answers` are reproducible
# inside the image as well as outside it. Not an installed package (see
# `[tool.setuptools.packages.find]` in pyproject.toml) -- plain source
# next to `app/`, run with `python -m` from this WORKDIR.
COPY evals ./evals
# Demo Mode (P3) and its committed, precomputed fixtures -- the same
# `demo.app:app` and `python -m demo.seed` this image can now serve and
# seed, so the deterministic keyless demo runs from this image as well as
# from a local checkout. Not an installed package, same reasoning as
# `evals/` above -- plain source, run with `python -m`/`uvicorn` from
# this WORKDIR. Adds no new dependency: `demo/` only imports packages
# `app/` already requires.
COPY demo ./demo

# CPU-only PyTorch (locked decision D6). `sentence-transformers`
# (pyproject.toml) declares `torch>=2.2` and is agnostic to which build
# satisfies it -- the local embedding and cross-encoder models this
# image serves are CPU inference only, so the CUDA build `pip install .`
# would otherwise resolve buys this image several gigabytes of unused
# GPU libraries. Installed *before* `pip install .` so the later,
# unconstrained `torch` requirement is already satisfied by this build
# and pip does not silently swap in a different one.
#
# **Not build-verified in this development environment.** This sandbox's
# own network policy blocks `download.pytorch.org` with the same
# CONNECT-tunnel refusal that blocks the HuggingFace weights this image
# also cannot bake in (see the README's milestone 10 section) -- so this
# line has been read against sentence-transformers' own declared
# constraint (`torch>=2.2`, confirmed compatible) but never actually run
# here. It is the standard, documented method for a CPU-only PyTorch
# install; if the index is unreachable at build time in a real
# environment, `pip install` fails loudly here and the build stops -- it
# never silently falls back to a different, unintended build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

RUN pip install . && rm -rf build knowledgeos.egg-info

# Storage for uploaded documents: absolute, and outside `/app`, so a
# volume mounted here survives an image rebuild. The default is a
# container-local path so the image still runs correctly with no volume
# configured at all -- see docker-compose.yml for the persistent mount.
ENV KNOWLEDGEOS_STORAGE_ROOT=/var/lib/knowledgeos/storage
RUN useradd --system --create-home --home-dir /home/knowledgeos --uid 10001 knowledgeos \
    && mkdir -p "${KNOWLEDGEOS_STORAGE_ROOT}" \
    && chown -R knowledgeos:knowledgeos "${KNOWLEDGEOS_STORAGE_ROOT}" /app /home/knowledgeos
ENV HOME=/home/knowledgeos
USER knowledgeos

EXPOSE 8000

# KNOWLEDGEOS_DATABASE_URL, KNOWLEDGEOS_LLM_MODEL, and the Gemini
# credential (GEMINI_API_KEY, or GOOGLE_API_KEY which takes precedence)
# must be supplied at run time -- see .env.example and the README's
# "Deployment" section. Nothing here is baked into the image: no key, no
# password, no literal connection string with real credentials anywhere
# in this file.
#
# `/health` (liveness, locked decision D13) -- deliberately not `/ready`.
# `/health` touches nothing (see app/api/health.py); `/ready` checks the
# database, the migration state, and the `vector` extension, and would
# have an orchestrator restart a perfectly healthy process merely
# because migrations have not been applied yet (see docker-compose.yml
# and app/main.py's own documented decision on why migrations are never
# run automatically).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Migrations are a separate, explicit step -- `docker compose run --rm
# app alembic upgrade head` -- never run from here or from any
# application startup path. See app/main.py's own documented decision:
# "Migrations are not run from here."
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
