"""
agent 서비스: 검색 API (public 망, 포트 8090)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.packages.agent import search

app = FastAPI(
    title="Product Search API (agent)",
    description="검색 API — OpenSearch에서 상품 코드 조회 (public)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)


@app.get("/")
async def root():
    return {
        "service": "agent",
        "message": "Product Search API (agent)",
        "docs": "/docs",
        "health": "/health",
        "search": "GET /api/v1/search?q=검색어&size=50",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agent"}
