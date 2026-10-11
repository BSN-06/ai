"""Database engine and session management."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings, BASE_DIR

import os

# Build absolute database path (SURVEILLANCE_DB_PATH overrides, used by the test suite)
db_path = os.environ.get("SURVEILLANCE_DB_PATH") or (BASE_DIR / "data" / "surveillance.db")
DATABASE_URL = f"sqlite:///{db_path}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency to provide a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _add_missing_columns(bind):
    """Minimal forward-only migration for SQLite: add columns introduced after a DB was created."""
    from sqlalchemy import inspect, text
    insp = inspect(bind)
    with bind.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in existing:
                    col_type = col.type.compile(dialect=bind.dialect)
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {col_type}'))


def init_db(bind=None):
    """Create all database tables (and add any newly introduced columns)."""
    from app.db import models as _  # noqa: F401 — ensure models are registered
    bind = bind or engine
    Base.metadata.create_all(bind=bind)
    _add_missing_columns(bind)
