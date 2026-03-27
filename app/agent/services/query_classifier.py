"""질문 유형 분류 — natural_language | noun (Bedrock LLM)."""
from app.core.config import get_settings
from app.core.logging import get_agent_logger
from app.core.prompt_loader import load_prompt

logger = get_agent_logger()

_DEFAULT = """You are a query classifier for a product search system. Classify the user query into one of: natural_language, noun. Output only the single word."""


def classify_query_type(query: str) -> str:
    if not query or not str(query).strip():
        return "natural_language"
    try:
        import boto3
        from botocore.exceptions import ClientError
        s = get_settings()
        client = boto3.client("bedrock-runtime", region_name=s.aws_region)
        prompt = load_prompt("agent", "query_classifier_system", fallback=_DEFAULT)
        response = client.converse(
            modelId=s.nova_reconstruct_model_id,
            messages=[{"role": "user", "content": [{"text": query}]}],
            system=[{"text": prompt}],
            inferenceConfig={"maxTokens": 16, "temperature": 0.1},
        )
        text = (
            response.get("output", {})
            .get("message", {})
            .get("content", [{}])[0]
            .get("text", "")
            .strip()
            .lower()
        )
        if text == "noun":
            logger.info("질문 유형 분류 query=%r -> noun", query)
            return "noun"
        logger.info("질문 유형 분류 query=%r -> natural_language", query)
        return "natural_language"
    except Exception as e:
        logger.warning("질문 유형 분류 오류: %s, 기본값 natural_language", e)
        return "natural_language"
