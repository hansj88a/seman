"""
agent 전용: 4단계 벡터 임베딩 — AWS Amazon Titan (검색 쿼리 임베딩)
"""
import json

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings
from app.logging_config import get_agent_logger

logger = get_agent_logger()


def get_bedrock_runtime():
    s = get_settings()
    return boto3.client("bedrock-runtime", region_name=s.aws_region)


def get_embedding(text: str) -> list[float]:
    """
    4단계: Titan v2로 텍스트 임베딩 벡터 반환.
    빈 문자열이면 빈 리스트.
    """
    if not (text and str(text).strip()):
        return []

    s = get_settings()
    client = get_bedrock_runtime()
    body = {
        "inputText": str(text).strip(),
        "dimensions": s.embed_dimensions,
        "normalize": True,
    }
    try:
        response = client.invoke_model(
            modelId=s.bedrock_embed_model_id,
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        vec = result.get("embedding", [])
        logger.info("벡터 임베딩 완료 text_len=%d vector_dim=%d", len(text), len(vec))
        return vec
    except ClientError as e:
        raise RuntimeError(f"Bedrock Titan embedding failed: {e}") from e
