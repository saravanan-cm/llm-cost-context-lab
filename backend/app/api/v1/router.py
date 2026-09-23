from fastapi import APIRouter

from app.api.v1.routes import chat, health, knowledge, usage

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(usage.router)
api_router.include_router(knowledge.router)
