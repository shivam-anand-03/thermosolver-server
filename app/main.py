"""FastAPI application factory and error handling."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router
from .config import settings
from .solver.utils import InputValidationError, SolverError

logger = logging.getLogger("thermosolve")


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="1.0.0",
                  description="1-D heat conduction solver: finite differences + hand-written Gaussian elimination.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)

    @app.exception_handler(InputValidationError)
    async def _input_error(_: Request, exc: InputValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"success": False, "error": exc.message,
                                                      "details": exc.details, "kind": "input"})

    @app.exception_handler(SolverError)
    async def _solver_error(_: Request, exc: SolverError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"success": False, "error": exc.message,
                                                      "details": exc.details, "kind": "numerical"})

    @app.exception_handler(RequestValidationError)
    async def _schema_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = []
        for e in exc.errors():
            loc = " → ".join(str(p) for p in e.get("loc", []) if p != "body")
            details.append(f"{loc}: {e.get('msg')}")
        return JSONResponse(status_code=422, content={"success": False, "error": "Invalid input: some fields are "
                                                      "missing or have the wrong type.", "details": details,
                                                      "kind": "schema"})

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error", exc_info=exc)
        return JSONResponse(status_code=500, content={"success": False, "kind": "internal",
                                                      "error": "Unexpected solver error. Please check your inputs.",
                                                      "details": []})

    return app


app = create_app()
