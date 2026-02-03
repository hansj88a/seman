"""
agent 전용: Amazon Nova Micro로 사용자 질문 유형 분류 (일반 명사 vs 자연어)
프롬프트: app/prompts/agent/query_classifier_system.txt (PROMPTS_DIR로 오버라이드 가능)
"""
import boto3
from botocore.exceptions import ClientError

from app.config import get_settings
from app.core.prompt_loader import load_prompt
from app.logging_config import get_agent_logger

logger = get_agent_logger()

# 분류 결과 타입
QUERY_TYPE_NOUN = "noun"
QUERY_TYPE_NATURAL = "natural_language"

_DEFAULT_CLASSIFIER_PROMPT = """You are a query classifier for a product search system.
Given the user's search input, answer ONLY one word:
- NOUN: if the input is a general noun or short keyword (e.g. product name, category, single/few terms like "노트북", "무선마우스", "키보드").
- NATURAL_LANGUAGE: if the input is a full sentence or question (e.g. "저렴한 노트북 추천해주세요", "무선으로 쓸 수 있는 마우스 있어?").
Reply with exactly NOUN or NATURAL_LANGUAGE, nothing else."""


def _get_bedrock_client():
    s = get_settings()
    return boto3.client("bedrock-runtime", region_name=s.aws_region)


def classify_query(query: str) -> str:
    """
    사용자 질문이 일반 명사인지 자연어인지 Nova Micro로 판단.
    반환: "noun" | "natural_language"
    """
    if not (query and str(query).strip()):
        return QUERY_TYPE_NATURAL

    s = get_settings()
    client = _get_bedrock_client()
    system_prompt = load_prompt("agent", "query_classifier_system", fallback=_DEFAULT_CLASSIFIER_PROMPT)

    user_message = f"User search input: {query.strip()}"

    try:
        response = client.converse(
            modelId=s.nova_classify_model_id,
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"maxTokens": 16, "temperature": 0.0, "topP": 0.2},
        )
        text = (
            response.get("output", {})
            .get("message", {})
            .get("content", [{}])[0]
            .get("text", "")
            .strip()
            .upper()
        )
        # NOUN / NATURAL_LANGUAGE 추출
        if "NATURAL" in text:
            logger.info("질문 유형 분류 query=%r -> natural_language", query)
            return QUERY_TYPE_NATURAL
        logger.info("질문 유형 분류 query=%r -> noun", query)
        return QUERY_TYPE_NOUN
    except ClientError as e:
        logger.warning("Nova 분류 실패, 기본값 natural_language 사용: %s", e)
        return QUERY_TYPE_NATURAL
    except Exception as e:
        logger.warning("질문 유형 분류 오류: %s, 기본값 natural_language", e)
        return QUERY_TYPE_NATURAL
