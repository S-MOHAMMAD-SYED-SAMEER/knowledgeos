"""The declarative base every model inherits from.

The naming convention is the whole reason this module exists separately. Left
to itself, PostgreSQL names a constraint one way and SQLAlchemy's metadata
imagines another, so `alembic revision --autogenerate` produces a migration
that drops and recreates constraints which never changed. Naming them by rule
means the database and the metadata agree, and `alembic check` stays quiet
when nothing has actually moved.
"""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


__all__ = ["NAMING_CONVENTION", "Base"]
