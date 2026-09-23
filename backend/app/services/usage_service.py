"""Usage ledger and credit balance persistence."""

from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.errors import InsufficientCreditsError, UserNotFoundError
from app.db.models import UsageRecord, User


@dataclass(frozen=True)
class UsageEntry:
    user_id: str
    conversation_id: str
    provider: str
    model: str
    provider_request_id: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    credits_consumed: Decimal
    latency_ms: int


@dataclass(frozen=True)
class UsageSummary:
    user_id: str
    total_credits: Decimal
    credits_used: Decimal
    credits_remaining: Decimal
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost: Decimal


class UsageService:
    def __init__(self, db: Session) -> None:
        self._db = db

    def ensure_user(self, user_id: str, initial_credits: Decimal) -> None:
        if self._db.get(User, user_id) is None:
            self._db.add(User(id=user_id, total_credits=initial_credits, credits_used=Decimal(0)))
            self._db.commit()

    def get_remaining_credits(self, user_id: str) -> Decimal:
        user = self._get_user(user_id)
        return user.total_credits - user.credits_used

    def ensure_can_afford(self, user_id: str, required_credits: Decimal) -> Decimal:
        """Raise ``InsufficientCreditsError`` unless the balance covers ``required_credits``."""
        remaining = self.get_remaining_credits(user_id)
        if remaining <= 0 or remaining < required_credits:
            raise InsufficientCreditsError()
        return remaining

    def record_usage(self, entry: UsageEntry) -> Decimal:
        """Insert the ledger row and deduct credits in one transaction; return the new balance.

        The deduction is a relative ``UPDATE ... SET credits_used = credits_used + :x`` so
        concurrent requests don't lose updates. A future version can add a
        ``WHERE total_credits - credits_used >= :x`` guard or a reservation step to make the
        pre-check and deduction strictly atomic.
        """
        try:
            self._db.add(UsageRecord(**entry.__dict__))
            result = self._db.execute(
                update(User)
                .where(User.id == entry.user_id)
                .values(credits_used=User.credits_used + entry.credits_consumed)
            )
            if result.rowcount != 1:
                raise UserNotFoundError()
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
        return self.get_remaining_credits(entry.user_id)

    def get_summary(self, user_id: str) -> UsageSummary:
        user = self._get_user(user_id)
        row = self._db.execute(
            select(
                func.count(UsageRecord.id),
                func.coalesce(func.sum(UsageRecord.input_tokens), 0),
                func.coalesce(func.sum(UsageRecord.output_tokens), 0),
                func.coalesce(func.sum(UsageRecord.total_tokens), 0),
                func.coalesce(func.sum(UsageRecord.total_cost), Decimal(0)),
            ).where(UsageRecord.user_id == user_id)
        ).one()
        return UsageSummary(
            user_id=user_id,
            total_credits=user.total_credits,
            credits_used=user.credits_used,
            credits_remaining=user.total_credits - user.credits_used,
            request_count=int(row[0]),
            total_input_tokens=int(row[1]),
            total_output_tokens=int(row[2]),
            total_tokens=int(row[3]),
            total_cost=row[4],
        )

    def _get_user(self, user_id: str) -> User:
        self._db.expire_all()
        user = self._db.get(User, user_id)
        if user is None:
            raise UserNotFoundError()
        return user
