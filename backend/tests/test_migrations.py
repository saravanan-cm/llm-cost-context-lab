from decimal import Decimal

from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app.db.models import User
from app.db.session import build_engine, run_migrations


def test_migrations_create_schema_and_store_exact_decimals(tmp_path):
    url = f"sqlite:///{tmp_path / 'test.db'}"

    run_migrations(url)

    engine = build_engine(url)
    assert {"users", "usage_records", "alembic_version"} <= set(inspect(engine).get_table_names())
    with Session(engine) as session:
        session.add(User(id="u", total_credits=Decimal("1000"), credits_used=Decimal("0.000000000001")))
        session.commit()
    with Session(engine) as session:
        assert session.get(User, "u").credits_used == Decimal("0.000000000001")
    engine.dispose()
