"""벡터 임베딩 — Bedrock Titan v2 (agent 검색용)."""
import json
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.core.config import get_settings
from app.core.logging import get_agent_logger

logger = get_agent_logger()


def get_embedding(text: str) -> Optional[list[float]]:
    if not text or not str(text).strip():
        return None
    s = get_settings()
    client = boto3.client("bedrock-runtime", region_name=s.aws_region)
    body = {
        "inputText": str(text).strip(),
        "dimensions": s.embed_dimensions,
        "normalize": True,
    }
    try:
        resp = client.invoke_model(
            modelId=s.bedrock_embed_model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        out = json.loads(resp["body"].read())
        vec = out.get("embedding")
        if vec:
            logger.info("벡터 임베딩 완료 text_len=%d vector_dim=%d", len(text), len(vec))
            return vec
    except ClientError as e:
        logger.warning("Bedrock 임베딩 실패: %s", e)
    except Exception as e:
        logger.warning("임베딩 오류: %s", e)
    return None
