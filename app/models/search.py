"""
검색 API 요청/응답 모델 (agent 서비스)
"""
from pydantic import BaseModel, Field


class SearchQueryParams(BaseModel):
    """검색 API 쿼리 파라미터"""

    q: str = Field(..., min_length=1, description="검색어")
    size: int = Field(50, ge=1, le=200, description="최대 결과 수")


class SearchResponse(BaseModel):
    """검색 API 응답"""

    query: str = Field(..., description="검색어")
    query_type: str = Field(
        ...,
        description="질문 유형: noun(일반 명사/키워드) | natural_language(자연어/문장). Nova Micro 분류 결과.",
    )
    product_codes: list[str] = Field(..., description="상품 코드 목록")
    count: int = Field(..., description="결과 개수")
