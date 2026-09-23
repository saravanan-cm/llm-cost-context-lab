from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.db.types import DecimalAmount


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    total_credits: Mapped[Decimal] = mapped_column(DecimalAmount, nullable=False)
    credits_used: Mapped[Decimal] = mapped_column(DecimalAmount, nullable=False, default=Decimal(0))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class UsageRecord(Base):
    """One row per successful LLM request (append-only ledger)."""

    __tablename__ = "usage_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    conversation_id: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    provider_request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    total_tokens: Mapped[int] = mapped_column(Integer)
    input_cost: Mapped[Decimal] = mapped_column(DecimalAmount)
    output_cost: Mapped[Decimal] = mapped_column(DecimalAmount)
    total_cost: Mapped[Decimal] = mapped_column(DecimalAmount)
    credits_consumed: Mapped[Decimal] = mapped_column(DecimalAmount)
    latency_ms: Mapped[int] = mapped_column(Integer)
