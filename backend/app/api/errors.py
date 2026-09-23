"""Map exceptions to clean JSON errors: {"error": {"code": ..., "message": ...}}.

Raw provider/database errors are logged server-side and never returned to clients.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.core.errors import AppError

logger = logging.getLogger(__name__)


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


async def _handle_app_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    log = logger.error if exc.status_code >= 500 else logger.info
    log("request_failed", extra={"path": request.url.path, "error_code": exc.code, "status": exc.status_code})
    return _error(exc.status_code, exc.code, exc.message)


async def _handle_db_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("database_error", extra={"path": request.url.path}, exc_info=exc)
    return _error(503, "database_unavailable", "The database is currently unavailable.")


async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_error", extra={"path": request.url.path}, exc_info=exc)
    return _error(500, "internal_error", "An unexpected error occurred.")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(SQLAlchemyError, _handle_db_error)
    app.add_exception_handler(Exception, _handle_unexpected)
