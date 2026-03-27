"""검색 API 요청·응답 모델."""
from pydantic import BaseModel, Field


class SearchQueryParams(BaseModel):
    q: str = Field(..., min_length=1, description="검색어")
    size: int = Field(default=50, ge=1, le=200, description="결과 수")


class SearchResponse(BaseModel):
    query: str = Field(..., description="재구성된 검색어")
    query_type: str = Field(..., description="knn 등")
    product_codes: list[str] = Field(default_factory=list)
    count: int = Field(..., ge=0)
