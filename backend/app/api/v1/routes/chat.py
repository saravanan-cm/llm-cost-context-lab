from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_chat_service, get_current_user_id
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import ErrorResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        402: {"model": ErrorResponse, "description": "Insufficient credits"},
        502: {"model": ErrorResponse, "description": "LLM provider error"},
        503: {"model": ErrorResponse, "description": "LLM or database unavailable"},
        504: {"model": ErrorResponse, "description": "LLM timeout"},
    },
)
def chat(
    request: ChatRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    # Sync route: FastAPI runs it in a worker thread, so the blocking SDK/DB calls
    # don't block the event loop.
    return service.reply(user_id, request)
