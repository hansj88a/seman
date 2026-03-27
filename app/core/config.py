"""
공통 설정 — 환경변수(.env) 기반.
"""
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenSearch
    opensearch_host: str = Field(default="http://localhost:9200", alias="OPENSEARCH_HOST")
    opensearch_index: str = Field(default="products", alias="OPENSEARCH_INDEX")
    opensearch_user: Optional[str] = Field(default=None, alias="OPENSEARCH_USER")
    opensearch_password: Optional[str] = Field(default=None, alias="OPENSEARCH_PASSWORD")
    opensearch_embedding_field: str = Field(default="embedding", alias="OPENSEARCH_EMBEDDING_FIELD")

    # S3 (manager)
    s3_bucket: Optional[str] = Field(default=None, alias="S3_BUCKET")
    s3_prefix: str = Field(default="products/", alias="S3_PREFIX")
    aws_region: str = Field(default="ap-northeast-2", alias="AWS_REGION")

    # Bedrock
    bedrock_embed_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        alias="BEDROCK_EMBED_MODEL_ID",
    )
    embed_dimensions: int = Field(default=1024, alias="EMBED_DIMENSIONS")
    nova_reconstruct_model_id: str = Field(
        default="amazon.nova-micro-v1:0",
        alias="NOVA_RECONSTRUCT_MODEL_ID",
    )

    # Agent
    morph_engine: Optional[str] = Field(default="kiwi", alias="MORPH_ENGINE")

    # Prompts
    prompts_dir: Optional[str] = Field(default=None, alias="PROMPTS_DIR")

    # Logging
    log_dir: str = Field(default="logs", alias="LOG_DIR")
    log_file_agent: str = Field(default="agent", alias="LOG_FILE_AGENT")
    log_file_manager: str = Field(default="manager", alias="LOG_FILE_MANAGER")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()
