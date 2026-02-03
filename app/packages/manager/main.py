"""
manager 서비스: 동기화 API (private 망, 포트 8091)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.packages.manager import sync

app = FastAPI(
    title="Product Sync API (manager)",
    description="동기화 API — S3 → OpenSearch 상품 등록 (private)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sync.router)


@app.get("/")
async def root():
    return {
        "service": "manager",
        "message": "Product Sync API (manager)",
        "docs": "/docs",
        "health": "/health",
        "sync": "POST /api/v1/sync/start",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "manager"}
