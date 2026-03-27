"""Agent 진입점 — 검색 API (public, 8090)."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent.routers import search
from app.core.logging import get_agent_logger

logger = get_agent_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Agent 서비스 시작 (검색 API)")
    yield
    logger.info("Agent 서비스 종료")


app = FastAPI(
    title="Product Search API",
    description="상품 시맨틱 검색 API (agent)",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(search.router, prefix="/api/v1", tags=["search"])
