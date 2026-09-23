from pydantic import BaseModel

from app.schemas.common import DecimalStr


class UsageSummaryResponse(BaseModel):
    user_id: str
    total_credits: DecimalStr
    credits_used: DecimalStr
    credits_remaining: DecimalStr
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost: DecimalStr
    currency: str = "USD"
