import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.api.errors import register_exception_handlers
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.errors import PricingNotConfiguredError
from app.core.logging import configure_logging
from app.core.request_context import RequestIdMiddleware
from app.db.session import build_engine, run_migrations
from app.services.pricing import PricingService
from app.services.usage_service import UsageService

logger = logging.getLogger(__name__)


def _startup(settings: Settings) -> None:
    if settings.db_auto_migrate:
        run_migrations(settings.database_url)
    engine = build_engine(settings.database_url)
    with Session(engine) as session:
        UsageService(session).ensure_user(settings.dev_user_id, settings.dev_user_initial_credits)
    engine.dispose()

    if settings.llm_provider == "openai":
        if settings.openai_api_key is None:
            logger.warning("OPENAI_API_KEY is not set; chat requests will fail until it is configured")
        try:
            PricingService().price_for(settings.openai_model)
        except PricingNotConfiguredError:
            logger.warning("No pricing configured for model", extra={"model": settings.openai_model})


def create_app(settings: Settings | None = None, *, run_startup: bool = True) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if run_startup:
            _startup(settings)
        logger.info("Started %s (%s)", settings.app_name, settings.environment)
        yield

    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
