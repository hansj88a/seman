"""동기화 API 응답 모델."""
from pydantic import BaseModel, Field


class SyncStartResponse(BaseModel):
    status: str = Field(..., description="completed 등")
    indexed: int = Field(..., ge=0)
    total_fetched: int = Field(..., ge=0)
    errors: list = Field(default_factory=list)
