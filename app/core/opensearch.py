"""
공통: OpenSearch 접속만 제공 (agent/manager 각자 검색·인덱싱 로직 보유)
"""
from opensearchpy import OpenSearch, RequestsHttpConnection

from app.config import get_settings


def get_client() -> OpenSearch:
    """OpenSearch 클라이언트 생성 (공통)"""
    s = get_settings()
    kwargs = {
        "hosts": [s.opensearch_host],
        "use_ssl": s.opensearch_host.startswith("https"),
        "verify_certs": True,
        "connection_class": RequestsHttpConnection,
    }
    if s.opensearch_user and s.opensearch_password:
        kwargs["http_auth"] = (s.opensearch_user, s.opensearch_password)
    return OpenSearch(**kwargs)
