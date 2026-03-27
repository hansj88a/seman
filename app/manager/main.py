"""Manager 진입점 — 동기화 API (private, 8091)."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.manager.routers import sync as sync_router
from app.core.logging import get_manager_logger

logger = get_manager_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Manager 서비스 시작 (동기화 API)")
    yield
    logger.info("Manager 서비스 종료")


app = FastAPI(
    title="Product Sync API",
    description="S3 → OpenSearch 동기화 API (manager)",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(sync_router.router, prefix="/api/v1", tags=["sync"])
