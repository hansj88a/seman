"""
agent 전용: 3단계 질문 재구성 — AWS Bedrock LLM 활용 (수정 가능)
프롬프트: app/prompts/agent/query_reconstructor_system.txt (PROMPTS_DIR로 오버라이드 가능)
"""
import boto3
from botocore.exceptions import ClientError

from app.config import get_settings
from app.core.prompt_loader import load_prompt
from app.logging_config import get_agent_logger

logger = get_agent_logger()

_DEFAULT_RECONSTRUCT_PROMPT = """You are a search query rewriter for a product search system.
Given the user's normalized query and key nouns extracted from it, output exactly one line: a concise search-optimized query for product search.
- Use only the most relevant product/category terms.
- Output in the same language as the input (Korean or English).
- Do not add explanations, only the single query line."""


def _get_bedrock_client():
    s = get_settings()
    return boto3.client("bedrock-runtime", region_name=s.aws_region)


def reconstruct_query(normalized_text: str, nouns: list[str]) -> str:
    """
    3단계: Bedrock LLM으로 검색에 적합한 질문 한 줄로 재구성.
    .env NOVA_RECONSTRUCT_MODEL_ID 로 모델 지정 가능.
    """
    if not normalized_text or not str(normalized_text).strip():
        return normalized_text or ""

    s = get_settings()
    model_id = s.nova_reconstruct_model_id
    client = _get_bedrock_client()

    nouns_str = ", ".join(nouns[:30]) if nouns else "(없음)"
    system_prompt = load_prompt("agent", "query_reconstructor_system", fallback=_DEFAULT_RECONSTRUCT_PROMPT)

    user_message = f"Normalized query: {normalized_text}\nKey nouns: {nouns_str}\nOutput one search-optimized query line:"

    try:
        response = client.converse(
            modelId=model_id,
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
