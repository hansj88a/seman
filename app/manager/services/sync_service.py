"""동기화 오케스트레이션: S3 → 임베딩 → OpenSearch."""
from app.core.logging import get_exception_location, get_manager_logger
from app.models import ProductIndexFields
from app.manager.services.embedding_client import get_embedding
from app.manager.services.opensearch_indexer import bulk_index_products
from app.manager.services.s3_client import fetch_products_from_s3

logger = get_manager_logger()


def _build_text_for_embedding(doc: dict, embedding_fields: list[str]) -> str:
    parts = []
    for field in embedding_fields:
        val = doc.get(field)
        if val is not None and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def _enrich_docs_with_embedding(docs: list[dict], index_fields: ProductIndexFields) -> list[dict]:
    vector_field = index_fields.embedding_vector_field
    embedding_fields = index_fields.embedding_fields
    total = len(docs)
    enriched = []
    for i, doc in enumerate(docs):
        text = _build_text_for_embedding(doc, embedding_fields)
        if text:
            try:
                vec = get_embedding(text)
                if vec:
                    doc = {**doc, vector_field: vec}
                else:
                    logger.warning("[2단계 임베딩] 빈 임베딩 반환 product_code=%s", doc.get("product_code"))
            except Exception as e:
                loc = get_exception_location(e)
                logger.warning("[2단계 임베딩] 실패 (위치: %s) product_code=%s: %s", loc, doc.get("product_code"), e)
        enriched.append(doc)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            logger.info("2단계 임베딩 진행: %d / %d 완료", i + 1, total)
    return enriched


def run_sync(index_fields: ProductIndexFields | None = None) -> dict:
    if index_fields is None:
        index_fields = ProductIndexFields()
    logger.info("동기화 시작 (임베딩 필드: %s)", index_fields.embedding_fields)

    try:
        logger.info("1단계 S3 상품 로드 실행")
        products = fetch_products_from_s3()
        total = len(products)
        logger.info("1단계 S3 상품 로드 완료: %d건", total)
    except Exception as e:
        loc = get_exception_location(e)
        logger.exception("[1단계 S3 상품 로드] 실패 (위치: %s): %s", loc, e)
        raise

    if not products:
        logger.warning("등록할 상품이 없습니다.")
        return {"status": "completed", "indexed": 0, "total_fetched": 0, "errors": []}

    try:
        logger.info("2단계 Titan v2 임베딩 적용 실행 (필드: %s)", index_fields.embedding_fields)
        docs = _enrich_docs_with_embedding(products, index_fields)
        logger.info("2단계 임베딩 적용 완료")
    except Exception as e:
        loc = get_exception_location(e)
        logger.exception("[2단계 임베딩 적용] 실패 (위치: %s): %s", loc, e)
        raise

    try:
        logger.info("3단계 OpenSearch bulk 인덱싱 실행")
        indexed, errors = bulk_index_products(docs, embedding_vector_field=index_fields.embedding_vector_field)
        logger.info("3단계 OpenSearch bulk 인덱싱 완료 indexed=%d", indexed)
    except Exception as e:
        loc = get_exception_location(e)
        logger.exception("[3단계 OpenSearch bulk 인덱싱] 실패 (위치: %s): %s", loc, e)
        raise

    if errors:
        for err in errors[:10]:
            logger.error("[3단계 인덱싱] 에러: %s", err)
        if len(errors) > 10:
            logger.error("[3단계 인덱싱] 외 %d건 추가 에러", len(errors) - 10)
    logger.info("동기화 완료: indexed=%d / total_fetched=%d, errors=%d", indexed, total, len(errors))
    return {"status": "completed", "indexed": indexed, "total_fetched": total, "errors": errors}
