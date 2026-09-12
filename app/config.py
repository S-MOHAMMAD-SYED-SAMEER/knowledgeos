"""Environment-driven configuration.

Settings arrive with the milestone that reads them: a chunk size nothing
chunks with, or a model name nothing calls, is a promise the code has not
made yet. Milestone 2 adds three, all of them about the upload boundary.

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


@lru_cache
def get_settings() -> Settings:
    """The one settings object, built on first use."""
    return Settings()


__all__ = ["DEFAULT_DATABASE_URL", "Settings", "get_settings"]
