"""OpenSearch 클라이언트 — agent/manager 공통."""
from functools import lru_cache
from typing import Any, Optional

from opensearchpy import OpenSearch

from app.core.config import get_settings


@lru_cache
def get_client() -> OpenSearch:
    s = get_settings()
    kwargs: dict[str, Any] = {
        "hosts": [s.opensearch_host],
        "use_ssl": s.opensearch_host.startswith("https"),
        "verify_certs": True,
    }
    if s.opensearch_user and s.opensearch_password:
        kwargs["http_auth"] = (s.opensearch_user, s.opensearch_password)
    return OpenSearch(**kwargs)
