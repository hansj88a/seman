"""
상품 인덱스 필드 설정: 임베딩 필드 vs 일반 텍스트 필드
OpenSearch 등록 시 Titan v2 임베딩 적용 대상과 일반 텍스트 컬럼 관리
"""
from pydantic import BaseModel, Field


class ProductIndexFields(BaseModel):
    """상품 문서 인덱스 필드 정의"""

    # Titan v2로 임베딩하여 벡터로 저장할 필드명 목록
    embedding_fields: list[str] = Field(
        default=["product_name", "description"],
        description="AWS Titan v2 임베딩 적용 필드 (벡터 컬럼으로 저장)",
    )
    # 일반 텍스트/키워드로만 저장할 필드명 목록 (검색·필터용)
    text_fields: list[str] = Field(
        default=["product_code", "product_name", "description", "keywords", "category", "price"],
        description="일반 텍스트/키워드 필드 (임베딩 미적용)",
    )
    # 임베딩 벡터가 저장될 단일 필드명 (OpenSearch knn_vector)
    embedding_vector_field: str = Field(
        default="embedding",
        description="OpenSearch에 저장되는 벡터 필드명",
    )
