"""
동기화 API 요청/응답 모델 (manager 서비스)
"""
from pydantic import BaseModel, Field


class SyncStartResponse(BaseModel):
    """동기화 시작 API 응답"""

    status: str = Field(..., description="처리 상태 (completed 등)")
    indexed: int = Field(..., description="OpenSearch에 등록된 건수")
    total_fetched: int = Field(..., description="S3에서 가져온 총 건수")
    errors: list[str] = Field(default_factory=list, description="에러 메시지 목록")
