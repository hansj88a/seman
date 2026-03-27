"""벡터 임베딩 — Bedrock Titan v2 (manager 동기화용)."""
import json
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.core.config import get_settings
from app.core.logging import get_manager_logger

logger = get_manager_logger()


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
            return vec
    except (ClientError, Exception) as e:
        logger.warning("임베딩 실패: %s", e)
    return None
