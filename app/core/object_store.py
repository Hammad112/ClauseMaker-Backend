"""Object storage provider.

Production: Cloudflare R2 (S3-compatible) via boto3.
Test: local filesystem at ./uploads/.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from loguru import logger

from app.core.config import get_settings


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> str: ...
    def get(self, key: str) -> bytes: ...


class _LocalObjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalObjectStore at {self.root}")

    def put(self, key: str, data: bytes, content_type: str) -> str:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()


class _R2ObjectStore:
    def __init__(self) -> None:
        import boto3  # type: ignore
        s = get_settings()
        self.bucket = s.r2_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=s.r2_endpoint,
            aws_access_key_id=s.r2_access_key,
            aws_secret_access_key=s.r2_secret_key,
            region_name=s.r2_region,
        )
        logger.info(f"R2ObjectStore bucket={self.bucket}")

    def put(self, key: str, data: bytes, content_type: str) -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return f"s3://{self.bucket}/{key}"

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()


_object_store_singleton: ObjectStore | None = None


def get_object_store() -> ObjectStore:
    global _object_store_singleton
    if _object_store_singleton is None:
        settings = get_settings()
        if settings.is_test_mode or not settings.r2_access_key:
            _object_store_singleton = _LocalObjectStore(Path("uploads"))
        else:
            _object_store_singleton = _R2ObjectStore()
    return _object_store_singleton


def storage_key(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return f"documents/{uuid4().hex}{ext}"
