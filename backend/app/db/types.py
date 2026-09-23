from decimal import Decimal
from typing import Any

from sqlalchemy import BigInteger, Numeric
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator, TypeEngine

from app.core.money import AMOUNT_SCALE, quantize_amount


class DecimalAmount(TypeDecorator[Decimal]):
    """Exact decimal amount (USD cost or credits) with 12 decimal places.

    PostgreSQL: native NUMERIC(38, 12).
    SQLite (no exact decimal type): a scaled BIGINT (value * 10**12), so storage and SQL
    SUM()/arithmetic stay exact. Range on SQLite is about +/-9.2 million units.
    """

    impl = BigInteger
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Numeric(38, AMOUNT_SCALE))
        return dialect.type_descriptor(BigInteger())

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        amount = quantize_amount(Decimal(value))
        if dialect.name == "postgresql":
            return amount
        return int(amount.scaleb(AMOUNT_SCALE))

    def process_result_value(self, value: Any, dialect: Dialect) -> Decimal | None:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return Decimal(value)
        return Decimal(int(value)).scaleb(-AMOUNT_SCALE)
