"""S3에서 상품 데이터 로드."""
import json

import boto3
from botocore.exceptions import ClientError

from app.core.config import get_settings


def get_s3_client():
    s = get_settings()
    return boto3.client("s3", region_name=s.aws_region)


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
