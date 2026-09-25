"""Environment-driven configuration.

Settings arrive with the milestone that reads them: a chunk size nothing
chunks with, or a model name nothing calls, is a promise the code has not
made yet. Milestone 2 added three about the upload boundary; milestone 3
adds four, about turning an uploaded file into chunks.

Every value comes from the environment under the `KNOWLEDGEOS_` prefix, or
from a `.env` file while developing. Nothing here is a secret yet — there is
no provider to hold a key for until generation arrives — and when there is
one, it comes from the environment and never from a default.

`get_settings` is cached so the object is built once and read everywhere, and
`cache_clear()` is what a test uses to change its mind. That is deliberate:
the alternative is a module-level instance built at import time, which cannot
be changed at all.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The local development database. Every developer's checkout points here
# until something says otherwise; nothing in this repository has ever pointed
# anywhere else.
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://knowledgeos:knowledgeos@localhost:5432/knowledgeos"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KNOWLEDGEOS_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "KnowledgeOS"
    environment: str = "local"
    # Off by default, so normal application behaviour is unchanged unless an
    # operator deliberately opts in. When true, `app/api/query.py::llm_provider`
    # resolves to `DemoLLMProvider` instead of the real Gemini adapter —
    # structurally, not by convention: nothing in this codebase imports
    # `app.providers.gemini_llm` from inside the demo path, so a demo
    # deployment cannot silently reach a real, billed model even if a Gemini
    # credential happens to be present in its environment. Embeddings and
    # reranking are untouched by this flag — demo mode still uses the real
    # local BGE and cross-encoder providers, per the project's own decision
    # that fake vectors are "meaningless" for anything that must resemble
    # real retrieval (see `app/providers/fake_embeddings.py`).
    demo_mode: bool = False
    # SQLAlchemy statement echoing. Off by default: echoed SQL carries whole
    # document titles and filenames into the log.
    debug: bool = False

    # psycopg 3 through SQLAlchemy 2.x, which is what the specification's
    # `postgresql+psycopg://` scheme means. PostgreSQL is the only datastore
    # this project has, so there is one URL and no second one to disagree
    # with it — Alembic reads this same setting.
    database_url: str = DEFAULT_DATABASE_URL

    # --- Uploads ---
    # Where uploaded files are kept. Relative paths resolve against the
    # project root, and `var/` is gitignored. The database stores paths
    # relative to this, so moving the root does not invalidate a row.
    storage_root: Path = Path("var/documents")

    # The specification requires a maximum upload size in configuration.
    # 25 MiB comfortably holds a scanned policy PDF and still refuses a body
    # sent to exhaust memory. Enforced while reading in bounded chunks, never
    # from the Content-Length header a client chose.
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, gt=0)

    # The specification requires an allowed-extension whitelist. These are
    # the four formats the parsers in milestone 3 will handle; an extension
    # accepted here that nothing can later parse would be a job that fails
    # after the caller has already been told the upload succeeded.
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".md", ".markdown", ".txt")

    # --- Chunking ---
    # The specification's terminology is "tokens", and these keep its names.
    # The unit this milestone counts is a **whitespace-delimited word**: no
    # tokenizer is specified anywhere, and the embedding model's own
    # tokenizer would mean downloading a model asset, which milestone 3 has
    # no business doing. See the README for the consequence at milestone 4.
    chunk_size_tokens: int = Field(default=512, gt=0)
    chunk_overlap_tokens: int = Field(default=64, ge=0)

    # --- Ingestion runner ---
    # How often the runner looks for queued work. Polling rather than an
    # in-process background task, so a job survives a restart.
    ingestion_poll_seconds: float = Field(default=5.0, gt=0)
    # How many times a job may be tried before it is left failed. The
    # specification requires retries to be "bounded by attempts" and names no
    # number, so the number lives here rather than in the code.
    max_attempts: int = Field(default=3, gt=0)

    # --- Retrieval ---
    # RRF's own constant. The specification fixes the formula and gives this
    # a default of 60 while calling it "configurable" — the only part of
    # retrieval's numbers it says that about. The per-channel candidate limit
    # (50) and the final count handed to reranking (20) are not settings:
    # the specification states them as facts about what "top 50" and "top
    # 20" mean, and a request-tunable version of either would make those
    # sentences untrue on demand.
    rrf_k: int = Field(default=60, gt=0)

    # The longest query `POST /query` accepts before answering 422. Nothing
    # in the specification sets a number; this exists so a very long query is
    # rejected loudly rather than silently truncated by the embedding
    # model's own token limit, which would embed a different query than the
    # one that was typed.
    query_max_length: int = Field(default=1000, gt=0)

    # --- Generation (milestone 8) ---
    # Deliberately no default. The generation provider for this build is
    # Google Gemini (a documented deviation from the specification's
    # Anthropic wording — see the README's milestone 8 section), and no
    # model ID is chosen from memory anywhere in this codebase: at the time
    # this setting was added, no Gemini credential was available in the
    # build environment to verify one against the live API. An operator who
    # has verified a model with `client.models.list()` sets it here; a
    # missing value is a config error the moment generation is attempted
    # (`app/providers/gemini_llm.py`), not a silently wrong guess.
    llm_model: str | None = None

    # The specification names no cap on generation length. This exists for
    # the same reason `query_max_length` does — an explicit limit rather
    # than whatever the provider's own default happens to be.
    llm_max_output_tokens: int = Field(default=2048, gt=0)

    # The specification's own words: the abstention threshold "is calibrated
    # on a dev split of the eval questions, not chosen by taste." No
    # calibration has been run in this build (it needs the real local
    # cross-encoder, absent from this environment's model cache — the same
    # condition milestones 4, 6 and 7 already documented). `None` means the
    # rerank-score abstention trigger is inactive; only a genuinely
    # calibrated run may set this, and the calibration must be documented
    # when it does.
    abstention_rerank_threshold: float | None = None

    # --- Observability / cost (milestone 9) ---
    # `{"<model name>": {"input": <usd per 1M tokens>, "output": <usd per
    # 1M tokens>}}`. Empty by default: the specification requires unknown
    # pricing to raise a config error rather than silently become zero
    # (§14), and no model's real, published rate has been verified and
    # entered here — inventing one for an unverified Gemini model is
    # exactly what this default refuses to do. An operator adds an entry
    # only once they have confirmed the vendor's current rate.
    llm_pricing_usd_per_million_tokens: dict[str, dict[str, float]] = Field(
        default_factory=dict
    )


@lru_cache
def get_settings() -> Settings:
    """The one settings object, built on first use."""
    return Settings()


__all__ = ["DEFAULT_DATABASE_URL", "Settings", "get_settings"]
