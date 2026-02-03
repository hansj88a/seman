"""
검색 API: 검색어 → OpenSearch에서 상품 코드 조회 (agent 서비스)
"""
from fastapi import APIRouter, Depends, HTTPException

from app.models.search import SearchQueryParams, SearchResponse
from app.packages.agent.services.search_service import search_product_codes

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.get(
    "/search",
    summary="상품 검색",
    description="검색어를 입력하면 OpenSearch에서 매칭되는 상품 코드 목록을 반환합니다.",
    response_model=SearchResponse,
)
async def search_products(params: SearchQueryParams = Depends(SearchQueryParams)) -> SearchResponse:
    try:
        product_codes, query_type, normalized_query = search_product_codes(
            query=params.q, size=params.size
        )
        return SearchResponse(
            query=normalized_query,
            query_type=query_type,
            product_codes=product_codes,
            count=len(product_codes),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"검색 실패: {str(e)}") from e
