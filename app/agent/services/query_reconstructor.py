"""질문 재구성 — Bedrock LLM으로 검색용 한 줄로 재구성."""
import boto3
from botocore.exceptions import ClientError

from app.core.config import get_settings
from app.core.logging import get_agent_logger
from app.core.prompt_loader import load_prompt

logger = get_agent_logger()

_DEFAULT = """You are a search query rewriter. Given normalized query and key nouns, output exactly one line: a concise search-optimized query. Same language as input. No explanations."""


def reconstruct_query(normalized_text: str, nouns: list[str]) -> str:
    if not normalized_text or not str(normalized_text).strip():
        return normalized_text or ""
    s = get_settings()
    client = boto3.client("bedrock-runtime", region_name=s.aws_region)
    nouns_str = ", ".join(nouns[:30]) if nouns else "(없음)"
    system_prompt = load_prompt("agent", "query_reconstructor_system", fallback=_DEFAULT)
    user_message = f"Normalized query: {normalized_text}\nKey nouns: {nouns_str}\nOutput one search-optimized query line:"
    try:
        response = client.converse(
            modelId=s.nova_reconstruct_model_id,
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"maxTokens": 128, "temperature": 0.2, "topP": 0.9},
        )
        text = (
            response.get("output", {})
            .get("message", {})
            .get("content", [{}])[0]
            .get("text", "")
            .strip()
        )
        if text:
            logger.info("질문 재구성 normalized=%r -> reconstructed=%r", normalized_text, text)
            return text
    except ClientError as e:
        logger.warning("질문 재구성 LLM 실패, 원문 사용: %s", e)
    except Exception as e:
        logger.warning("질문 재구성 오류, 원문 사용: %s", e)
    return normalized_text.strip()
