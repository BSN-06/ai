"""
Pytest configuration and fixtures for the surveillance system test suite.

The whole application (API dependency, background workers, replay processor) is pointed at an
isolated SQLite file through SURVEILLANCE_DB_PATH *before* the app is imported.
"""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="surveillance_test_")
os.environ["SURVEILLANCE_DB_PATH"] = os.path.join(_TMP, "test_surveillance.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db.database import Base, engine, SessionLocal, init_db  # noqa: E402
from app.main import app  # noqa: E402

TestingSessionLocal = SessionLocal


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create test database tables once per session."""
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session():
    """Provide a database session per test function, rolled back afterwards."""
    connection = engine.connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="session")
def client():
    """FastAPI test client (lifespan runs: tables + seed data)."""
    with TestClient(app) as c:
        yield c
