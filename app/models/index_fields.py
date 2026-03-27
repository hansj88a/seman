"""OpenSearch 인덱스·임베딩 필드 정의 — manager 동기화용."""
from pydantic import BaseModel, Field


class ProductIndexFields(BaseModel):
    embedding_fields: list[str] = Field(
        default_factory=lambda: ["product_name", "description", "keywords", "category"],
        description="임베딩 생성에 사용할 필드",
    )
    text_fields: list[str] = Field(
        default_factory=lambda: ["product_name", "description", "keywords", "category"],
        description="일반 텍스트 필드",
    )
    embedding_vector_field: str = Field(default="embedding", description="벡터 필드명")
