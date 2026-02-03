"""
agent 전용: 상품 검색 5단계 프로세스
1. 텍스트 정규화 (Unicode, 특수문자 제거)
2. 형태소 분석 (Kiwi/MeCab 명사 추출)
3. 질문 재구성 (AWS Bedrock LLM, 수정 가능)
4. 벡터 임베딩 (AWS Amazon Titan)
5. OpenSearch k-NN 검색
"""
from app.config import get_settings
from app.core.opensearch import get_client
from app.logging_config import get_agent_logger, get_exception_location
from app.packages.agent.services.embedding_client import get_embedding
from app.packages.agent.services.morph_analyzer import extract_nouns
from app.packages.agent.services.query_reconstructor import reconstruct_query
from app.packages.agent.services.text_normalizer import step1_normalize

logger = get_agent_logger()



def _knn_search(client, index: str, vector: list[float], size: int, vector_field: str) -> list[str]:
    """OpenSearch k-NN 검색으로 상품 코드 목록 반환."""
    body = {
        "size": size,
        "_source": ["product_code"],
        "query": {
            "knn": {
                vector_field: {
                    "vector": vector,
                    "k": size,
                }
            }
        },
    }
    response = client.search(index=index, body=body)
    hits = response.get("hits", {}).get("hits", [])
    return [h["_source"].get("product_code", "") for h in hits if h.get("_source")]


def search_product_codes(query: str, size: int = 50) -> tuple[list[str], str, str]:
    """
    5단계 상품 검색.
    반환: (상품 코드 목록, query_type "knn", 최종 검색에 사용된 질문 텍스트)
    """
    logger.info("검색 요청 시작 query=%r size=%s", query, size)
    client = get_client()
    settings = get_settings()
    index = settings.opensearch_index
    vector_field = settings.opensearch_embedding_field

    # 1단계: 텍스트 정규화 (Unicode, 특수문자 제거)
    try:
        logger.info("1단계 텍스트 정규화 실행")
        step1_text = step1_normalize(query)
        if not step1_text:
            step1_text = query.strip() or query
        logger.info("1단계 텍스트 정규화 완료 original=%r -> normalized=%r", query, step1_text)
    except Exception as e:
        loc = get_exception_location(e)
        logger.exception("[1단계 텍스트 정규화] 실패 (위치: %s): %s", loc, e)
        raise

    # 2단계: 형태소 분석 (Kiwi/MeCab 명사 추출)
    try:
        logger.info("2단계 형태소 분석(명사) 실행")
        nouns = extract_nouns(step1_text)
        logger.info("2단계 형태소 분석(명사) 완료 nouns=%s", nouns[:15])
    except Exception as e:
        loc = get_exception_location(e)
        logger.exception("[2단계 형태소 분석] 실패 (위치: %s): %s", loc, e)
        raise

    # 3단계: 질문 재구성 (Bedrock LLM)
    try:
        logger.info("3단계 질문 재구성 실행")
        reconstructed = reconstruct_query(step1_text, nouns)
        if not reconstructed:
            reconstructed = step1_text
        logger.info("3단계 질문 재구성 완료 reconstructed=%r", reconstructed)
    except Exception as e:
        loc = get_exception_location(e)
        logger.exception("[3단계 질문 재구성] 실패 (위치: %s): %s", loc, e)
        raise

    # 4단계: 벡터 임베딩 (Titan)
    try:
        logger.info("4단계 벡터 임베딩 실행")
        vector = get_embedding(reconstructed)
        if not vector:
            logger.warning("4단계 임베딩 실패 또는 빈 벡터, 텍스트 검색 fallback 불가 시 빈 결과 반환")
            return [], "knn", reconstructed
        logger.info("4단계 벡터 임베딩 완료")
    except Exception as e:
        loc = get_exception_location(e)
        logger.exception("[4단계 벡터 임베딩] 실패 (위치: %s): %s", loc, e)
        raise

    # 5단계: OpenSearch k-NN 검색
    try:
        logger.info("5단계 OpenSearch k-NN 검색 실행 index=%s size=%s", index, size)
        codes = _knn_search(client, index, vector, size, vector_field)
        logger.info("5단계 k-NN 검색 완료 결과 수=%d", len(codes))
        logger.info("상품 검색 완료 query=%r 결과 수=%d", reconstructed, len(codes))
        return codes, "knn", reconstructed
    except Exception as e:
        loc = get_exception_location(e)
        logger.exception("[5단계 k-NN 검색] 실패 (위치: %s) query=%r: %s", loc, reconstructed, e)
        raise
