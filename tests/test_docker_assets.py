"""Static verification of the Docker/CI assets (milestone 10).

**No build or runtime test exists here, deliberately.** The Docker daemon
is unavailable in this development environment (`docker info` fails:
"Cannot connect to the Docker daemon" -- confirmed, not assumed), so
nothing here claims to have built an image or run a container. Every test
below reads the committed files as text and checks the properties the
locked decisions require of them -- non-root user, healthcheck target,
absence of secrets, pgvector-capable database image, persistent volumes.
"""

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text()


def _compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())


def test_dockerfile_exists() -> None:
    assert (ROOT / "Dockerfile").is_file()


def test_docker_compose_exists() -> None:
    assert (ROOT / "docker-compose.yml").is_file()


def test_dockerignore_exists() -> None:
    assert (ROOT / ".dockerignore").is_file()


def test_ci_workflow_exists() -> None:
    assert (ROOT / ".github" / "workflows" / "ci.yml").is_file()


# --- Dockerfile ------------------------------------------------------------


def test_dockerfile_declares_a_non_root_user() -> None:
    source = _dockerfile()
    assert re.search(r"^USER\s+\S+", source, re.MULTILINE)
    assert "USER root" not in source
    assert "useradd" in source


def test_dockerfile_has_a_healthcheck_against_health_not_ready() -> None:
    source = _dockerfile()
    assert "HEALTHCHECK" in source
    healthcheck_block = source[source.index("HEALTHCHECK"):]
    assert "/health" in healthcheck_block.split("\n\n")[0]
    assert "/ready" not in healthcheck_block.split("\n\n")[0]


def test_dockerfile_exposes_port_8000() -> None:
    assert "EXPOSE 8000" in _dockerfile()


def test_dockerfile_uses_an_absolute_storage_root() -> None:
    match = re.search(r"KNOWLEDGEOS_STORAGE_ROOT=(\S+)", _dockerfile())
    assert match is not None
    assert match.group(1).startswith("/")


def test_dockerfile_runs_uvicorn_in_production_mode() -> None:
    source = _dockerfile()
    assert "uvicorn" in source
    assert "--reload" not in source


def test_dockerfile_never_runs_alembic() -> None:
    """Locked decision D8: migrations are a separate, explicit step,
    never baked into the image's build or start commands. The file may
    still *mention* the command in a comment explaining why it is
    deliberately absent from `RUN`/`CMD` -- this checks the executable
    instructions, not prose."""
    source = _dockerfile()
    executable_lines = [
        line for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    # `COPY alembic ./alembic` is expected and required -- the separate
    # migration step run against the built image needs the files present.
    # What must never appear in an executable instruction is *running* one.
    assert not any("alembic upgrade" in line for line in executable_lines)
    assert not any(re.search(r"\balembic\b", line) and "COPY" not in line for line in executable_lines)
    assert 'CMD ["uvicorn"' in source


def test_dockerfile_contains_no_secret_literal() -> None:
    source = _dockerfile().lower()
    for forbidden in ("api_key=", "password=", "gemini_api_key=\"", "secret="):
        assert forbidden not in source
    # The one literal credential-shaped strings in this file belong to
    # docker-compose.yml's local Postgres defaults, not here.
    assert "postgresql+psycopg://" not in _dockerfile()


def test_dockerfile_requests_cpu_only_torch_with_documentation() -> None:
    """Locked decision D6: CPU-only torch, verified compatible with
    sentence-transformers' own `torch>=2.2` constraint and documented as
    unverified-by-build in this environment -- never silently assumed to
    work."""
    source = _dockerfile()
    assert "download.pytorch.org/whl/cpu" in source
    assert "torch>=2.2" in source
    assert "build-verified" in source.lower()


# --- docker-compose.yml ------------------------------------------------


def test_compose_has_an_app_and_a_database_service() -> None:
    compose = _compose()
    assert "app" in compose["services"]
    assert "db" in compose["services"]


def test_compose_database_is_pgvector_capable() -> None:
    """Locked decision D7: never plain `postgres:16`."""
    image = _compose()["services"]["db"]["image"]
    assert "pgvector" in image
    assert image != "postgres:16"


def test_compose_has_persistent_volumes_for_database_and_storage() -> None:
    compose = _compose()
    volumes = compose.get("volumes", {})
    assert len(volumes) >= 2

    db_volumes = compose["services"]["db"].get("volumes", [])
    assert any("/var/lib/postgresql/data" in v for v in db_volumes)

    app_volumes = compose["services"]["app"].get("volumes", [])
    assert any("/var/lib/knowledgeos/storage" in v for v in app_volumes)


def test_compose_app_depends_on_database() -> None:
    depends_on = _compose()["services"]["app"]["depends_on"]
    assert "db" in depends_on


def test_compose_contains_no_literal_external_credential() -> None:
    """The local Postgres user/password are development defaults for a
    throwaway container, not a real secret -- but nothing naming a real
    provider credential (a Gemini key, a production database password)
    may appear as a literal value."""
    source = (ROOT / "docker-compose.yml").read_text()
    for forbidden in ("GEMINI_API_KEY:", "GOOGLE_API_KEY:", "KNOWLEDGEOS_LLM_MODEL:"):
        line = next(l for l in source.splitlines() if forbidden in l)
        assert "${" in line, f"{forbidden} is not read from the environment: {line!r}"


def test_compose_migration_step_is_documented_not_automatic() -> None:
    """Locked decision D8: `docker compose run --rm app alembic upgrade
    head` is documented as a separate command, and the app service's own
    command is never alembic."""
    source = (ROOT / "docker-compose.yml").read_text()
    assert "alembic upgrade head" in source  # documented in a comment
    compose = _compose()
    assert "command" not in compose["services"]["app"] or "alembic" not in str(
        compose["services"]["app"].get("command", "")
    )


# --- .dockerignore -----------------------------------------------------


def test_dockerignore_excludes_secrets_and_local_state() -> None:
    source = (ROOT / ".dockerignore").read_text()
    required = [".env", ".venv", "__pycache__", "var/", ".git", ".pytest_cache"]
    for entry in required:
        assert entry in source, f"{entry!r} is not excluded"


def test_dockerignore_keeps_env_example() -> None:
    source = (ROOT / ".dockerignore").read_text()
    assert "!.env.example" in source


# --- CI ------------------------------------------------------------------


def _ci_workflow() -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())


def test_ci_uses_pgvector_capable_postgres() -> None:
    workflow = _ci_workflow()
    job = workflow["jobs"]["test"]
    image = job["services"]["postgres"]["image"]
    assert "pgvector" in image


def test_ci_installs_dependencies_and_runs_the_suite() -> None:
    source = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "pip install" in source
    assert re.search(r"run:\s*pytest\b", source)


def test_ci_never_sets_an_external_provider_api_key() -> None:
    source = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    for forbidden in ("ANTHROPIC_API_KEY:", "GEMINI_API_KEY:", "GOOGLE_API_KEY:", "OPENAI_API_KEY:"):
        assert forbidden not in source


def test_ci_forces_huggingface_offline_mode() -> None:
    """Locked decision D9: CI must never download HuggingFace model
    weights -- the deterministic way to guarantee that regardless of
    whether the runner happens to have internet access."""
    source = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "HF_HUB_OFFLINE" in source
    assert "TRANSFORMERS_OFFLINE" in source


def test_ci_has_no_deployment_or_release_step() -> None:
    """Locked decision D9: keep CI focused."""
    workflow = _ci_workflow()
    job_names = {name.lower() for name in workflow["jobs"]}
    for forbidden in ("deploy", "release", "publish"):
        assert not any(forbidden in name for name in job_names)
