"""OpenSearch 인덱스 생성 및 bulk 인덱싱."""
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from app.core.config import get_settings
from app.core.opensearch import get_client


def ensure_index(
    client: OpenSearch,
    index: str,
    *,
    embedding_dimension: int = 1024,
    embedding_vector_field: str = "embedding",
) -> None:
    if not client.indices.exists(index=index):
        properties = {
            "product_code": {"type": "keyword"},
            "product_name": {"type": "text", "analyzer": "standard"},
            "description": {"type": "text", "analyzer": "standard"},
            "keywords": {"type": "text", "analyzer": "standard"},
            "price": {"type": "float"},
            "category": {"type": "keyword"},
            embedding_vector_field: {
                "type": "knn_vector",
                "dimension": embedding_dimension,
                "method": {
                    "name": "hnsw",
                    "space_type": "l2",
                    "engine": "faiss",
                    "parameters": {"ef_construction": 128, "m": 24},
                },
            },
        }
        client.indices.create(
            index=index,
            body={
                "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index": {"knn": True}},
                "mappings": {"properties": properties},
            },
        )


def bulk_index_products(
    docs: list[dict],
    *,
    embedding_dimension: int | None = None,
    embedding_vector_field: str = "embedding",
) -> tuple[int, list[str]]:
    client = get_client()
    settings = get_settings()
    index = settings.opensearch_index
    dim = embedding_dimension if embedding_dimension is not None else settings.embed_dimensions
    ensure_index(client, index, embedding_dimension=dim, embedding_vector_field=embedding_vector_field)
    actions = []
    for doc in docs:
        product_code = doc.get("product_code") or doc.get("id")
        if not product_code:
            continue
        actions.append({"_index": index, "_id": str(product_code), "_source": doc})
    errors = []
    success = 0
    if actions:
        try:
            ok, failed = bulk(client, actions, raise_on_error=False, raise_on_exception=False)
            success = ok
            for item in failed if isinstance(failed, list) else []:
                err = item.get("index", {}).get("error", {})
                if err:
                    errors.append(err.get("reason", str(err)))
        except Exception as e:
            errors.append(str(e))
    return success, errors
