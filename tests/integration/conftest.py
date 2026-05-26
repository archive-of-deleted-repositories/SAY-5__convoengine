"""Integration test fixtures backed by testcontainers Postgres."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

from convoengine.store import create_all, make_engine, make_session_factory


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    if os.environ.get("CONVOENGINE_TEST_DB_URL"):
        # Allow CI to pass in a pre-provisioned database via env.
        yield os.environ["CONVOENGINE_TEST_DB_URL"]
        return
    with PostgresContainer("postgres:16-alpine") as pg:
        raw_url = pg.get_connection_url()
        # testcontainers returns a psycopg2-style URL; rewrite to psycopg v3.
        url = raw_url.replace("postgresql+psycopg2", "postgresql+psycopg")
        if "+psycopg" not in url:
            url = url.replace("postgresql://", "postgresql+psycopg://")
        yield url


@pytest.fixture
def session_factory(postgres_url: str) -> Iterator[sessionmaker[Session]]:
    engine = make_engine(postgres_url)
    create_all(engine)
    factory = make_session_factory(engine)
    try:
        yield factory
    finally:
        # Drop tables between tests for isolation.
        from convoengine.store import Base

        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    s = session_factory()
    try:
        yield s
    finally:
        s.close()
