from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user_id, get_usage_service
from app.schemas.usage import UsageSummaryResponse
from app.services.usage_service import UsageService

router = APIRouter(tags=["usage"])


@router.get("/usage", response_model=UsageSummaryResponse)
def usage(
    user_id: Annotated[str, Depends(get_current_user_id)],
    service: Annotated[UsageService, Depends(get_usage_service)],
) -> UsageSummaryResponse:
    return UsageSummaryResponse(**asdict(service.get_summary(user_id)))
