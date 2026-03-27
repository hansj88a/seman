"""S3에서 상품 데이터 로드."""
import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Optional
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from app.core.config import get_settings

_S3_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
)


def get_s3_client():
    s = get_settings()
    return boto3.client("s3", region_name=s.aws_region)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """
    s3://bucket/prefix/... → (bucket, key_prefix_without_leading_slash).
    """
    raw = uri.strip()
    if not raw.lower().startswith("s3://"):
        raise ValueError(f"S3 URI가 아닙니다: {uri!r} (예: s3://bucket/folder/)")
    parsed = urlparse(raw)
    if not parsed.netloc:
        raise ValueError(f"S3 URI에 버킷이 없습니다: {uri!r}")
    path = (parsed.path or "").lstrip("/")
    return parsed.netloc, path


def normalize_s3_key_prefix(prefix: str) -> str:
    p = prefix.strip().lstrip("/")
    if not p:
        return ""
    return p if p.endswith("/") else f"{p}/"


def list_s3_object_keys_v2(
    bucket: str,
    prefix: str,
    *,
    client: Optional[object] = None,
) -> list[str]:
    """
    `list_objects_v2`로 prefix 하위 객체 키를 모두 반환(페이지네이션 포함).

    각 응답에서 `Contents`의 `Key`만 사용:

        [item["Key"] for item in response.get("Contents", []) if item.get("Key")]
    """
    c = client or get_s3_client()
    pfx = normalize_s3_key_prefix(prefix)
    keys: list[str] = []
    request: dict = {"Bucket": bucket, "Prefix": pfx}

    while True:
        response = c.list_objects_v2(**request)
        contents = response.get("Contents") or []
        keys.extend(
            item["Key"] for item in contents if item.get("Key")
        )
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
        if not token:
            break
        request = {
            "Bucket": bucket,
            "Prefix": pfx,
            "ContinuationToken": token,
        }

    return keys


def list_s3_image_keys(
    bucket: str,
    prefix: str,
    *,
    client: Optional[object] = None,
) -> list[str]:
    """`list_s3_object_keys_v2` 결과 중 이미지 확장자만 골라 키 이름순으로 반환."""
    c = client or get_s3_client()
    raw_keys = list_s3_object_keys_v2(bucket, prefix, client=c)
    keys: list[str] = []
    for key in raw_keys:
        if key.endswith("/"):
            continue
        low = key.lower()
        if any(low.endswith(ext) for ext in _S3_IMAGE_SUFFIXES):
            keys.append(key)
    return sorted(keys, key=lambda k: k.lower())


def get_s3_object_bytes(
    bucket: str,
    key: str,
    *,
    client: Optional[object] = None,
) -> bytes:
    c = client or get_s3_client()
    try:
        resp = c.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()
    except ClientError as e:
        raise ValueError(f"S3 get_object 실패: s3://{bucket}/{key} — {e}") from e


def download_s3_keys_to_directory(
    bucket: str,
    keys: Sequence[str],
    dest_dir: Path,
    *,
    client: Optional[object] = None,
) -> list[Path]:
    """
    지정한 키 순서대로 바이트를 받아 `dest_dir`에 저장하고, 저장된 로컬 경로 목록을 반환.
    동일 파일명 충돌 시 `name_2.ext` 형태로 회피.
    """
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    c = client or get_s3_client()
    written: list[Path] = []
    taken: set[str] = set()

    for key in keys:
        base = PurePosixPath(key).name
        if not base:
            continue
        candidate = dest_dir / base
        if candidate.name in taken or candidate.exists():
            stem = Path(base).stem
            suf = Path(base).suffix
            n = 1
            while True:
                alt = dest_dir / f"{stem}_{n}{suf}"
                if alt.name not in taken and not alt.exists():
                    candidate = alt
                    break
                n += 1
        taken.add(candidate.name)
        candidate.write_bytes(get_s3_object_bytes(bucket, key, client=c))
        written.append(candidate)
    return written


def download_s3_prefix_to_directory(
    bucket: str,
    prefix: str,
    dest_dir: Path,
    *,
    client: Optional[object] = None,
) -> list[Path]:
    """prefix 아래 이미지를 모두 내려받아 경로 목록 반환(키 정렬 순)."""
    keys = list_s3_image_keys(bucket, prefix, client=client)
    if not keys:
        return []
    return download_s3_keys_to_directory(bucket, keys, dest_dir, client=client)


def list_product_keys(bucket: str, prefix: str) -> list[str]:
    client = get_s3_client()
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key")
            if key and (key.endswith(".json") or key.endswith(".jsonl")):
                keys.append(key)
    return keys


def load_json_from_s3(bucket: str, key: str) -> list[dict]:
    client = get_s3_client()
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read().decode("utf-8")
    except ClientError as e:
        raise ValueError(f"S3 get_object failed: {key} - {e}") from e
    stripped = body.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        return json.loads(body)
    docs = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            docs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return docs


def _normalize_product_doc(doc: dict) -> dict:
    return {
        "product_code": doc.get("product_code") or doc.get("id") or doc.get("sku", ""),
        "product_name": doc.get("product_name") or doc.get("name") or doc.get("title", ""),
        "description": doc.get("description") or "",
        "keywords": doc.get("keywords") or "",
        "price": doc.get("price"),
        "category": doc.get("category") or doc.get("categories"),
    }


def fetch_products_from_s3() -> list[dict]:
    s = get_settings()
    if not s.s3_bucket:
        raise ValueError("S3_BUCKET이 설정되지 않았습니다.")
    bucket = s.s3_bucket
    prefix = (s.s3_prefix or "").rstrip("/") + "/"
    keys = list_product_keys(bucket, prefix)
    if not keys:
        raise ValueError(f"S3에 상품 파일이 없습니다: s3://{bucket}/{prefix}*")
    all_docs = []
    seen_ids = set()
    for key in keys:
        docs = load_json_from_s3(bucket, key)
        for doc in docs:
            pid = doc.get("product_code") or doc.get("id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_docs.append(_normalize_product_doc(doc))
    return all_docs
