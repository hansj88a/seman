"""
환경 설정 (OpenSearch, S3)
.env 또는 환경변수로 설정
"""
import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # OpenSearch
    opensearch_host: str = os.getenv("OPENSEARCH_HOST", "http://localhost:9200")
    opensearch_index: str = os.getenv("OPENSEARCH_INDEX", "products")
    opensearch_embedding_field: str = os.getenv("OPENSEARCH_EMBEDDING_FIELD", "embedding")
    opensearch_user: str = os.getenv("OPENSEARCH_USER", "")
    opensearch_password: str = os.getenv("OPENSEARCH_PASSWORD", "")

    # S3 (동기화 소스)
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_prefix: str = os.getenv("S3_PREFIX", "products/")  # 특정 경로 prefix
    aws_region: str = os.getenv("AWS_REGION", "ap-northeast-2")

    # AWS Bedrock Titan v2 임베딩
    bedrock_embed_model_id: str = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
    embed_dimensions: int = int(os.getenv("EMBED_DIMENSIONS", "1024"))

    # AWS Bedrock Nova Micro (agent: 질문 유형 분류, 질문 재구성 LLM)
    nova_classify_model_id: str = os.getenv("NOVA_CLASSIFY_MODEL_ID", "amazon.nova-micro-v1:0")
    nova_reconstruct_model_id: str = os.getenv("NOVA_RECONSTRUCT_MODEL_ID", "amazon.nova-micro-v1:0")

    # 형태소 분석 (agent: Kiwi / MeCab 명사 추출)
    morph_engine: str = os.getenv("MORPH_ENGINE", "kiwi")  # kiwi | mecab

    # 프롬프트 파일 경로 (프로덕션: 외부 경로 지정 가능, 미설정 시 앱 내장 prompts 사용)
    prompts_dir: str = os.getenv("PROMPTS_DIR", "")

    # 로그 (agent / manager 공통: 콘솔 + 파일, 파일명 service.yyyy-mm-dd.log)
    log_dir: str = os.getenv("LOG_DIR", "logs")
    log_file_agent: str = os.getenv("LOG_FILE_AGENT", "agent")  # → agent.yyyy-mm-dd.log
    log_file_manager: str = os.getenv("LOG_FILE_MANAGER", "manager")  # → manager.yyyy-mm-dd.log
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
