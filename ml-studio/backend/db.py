"""
Storage abstraction: Supabase when env vars are set, local filesystem otherwise.

Cloud path  → SUPABASE_URL + SUPABASE_SERVICE_KEY must be in .env
Local path  → data/uploads/<id>/  +  data/datasets.json

All public functions have the same signature regardless of which backend is active.
"""

import io
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

_client = None

# ── Local fallback paths ──────────────────────────────────────────────────────

_LOCAL_DIR = Path("data/uploads")
_LOCAL_META = Path("data/datasets.json")


def _local_meta() -> dict:
    if _LOCAL_META.exists():
        return json.loads(_LOCAL_META.read_text())
    return {}


def _save_local_meta(meta: dict) -> None:
    _LOCAL_META.parent.mkdir(parents=True, exist_ok=True)
    _LOCAL_META.write_text(json.dumps(meta, default=str, indent=2))


# ── Cloud helpers ─────────────────────────────────────────────────────────────

def is_cloud() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)


def _sb():
    global _client
    if _client is None:
        from supabase import create_client  # lazy import — only needed in cloud mode
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _client


# ── Public API ────────────────────────────────────────────────────────────────

def upsert_dataset(dataset_id: str, metadata: dict) -> None:
    """Persist dataset metadata (upsert)."""
    if is_cloud():
        _sb().table("datasets").upsert({"id": dataset_id, **metadata}).execute()
    else:
        meta = _local_meta()
        meta[dataset_id] = {**metadata, "id": dataset_id}
        _save_local_meta(meta)


def get_dataset(dataset_id: str) -> dict | None:
    """Return metadata dict or None if not found."""
    if is_cloud():
        resp = _sb().table("datasets").select("*").eq("id", dataset_id).maybe_single().execute()
        return resp.data
    else:
        return _local_meta().get(dataset_id)


def list_datasets() -> list[dict]:
    """Return all dataset metadata records."""
    if is_cloud():
        resp = _sb().table("datasets").select("*").order("created_at", desc=True).execute()
        return resp.data or []
    else:
        return list(_local_meta().values())


def upload_csv(dataset_id: str, content: bytes, filename: str = "data.csv") -> str:
    """
    Upload CSV bytes; return the storage path (cloud key or local file path).
    The path is what you pass back to download_df() later.
    """
    if is_cloud():
        storage_path = f"{dataset_id}/{filename}"
        bucket = _sb().storage.from_("datasets")
        # Remove existing file silently before re-uploading
        try:
            bucket.remove([storage_path])
        except Exception:
            pass
        bucket.upload(storage_path, content, {"content-type": "text/csv"})
        return storage_path
    else:
        dir_path = _LOCAL_DIR / dataset_id
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / filename
        file_path.write_bytes(content)
        return str(file_path)


def download_df(storage_path: str) -> pd.DataFrame:
    """Download a stored CSV and return it as a DataFrame."""
    if is_cloud():
        raw = _sb().storage.from_("datasets").download(storage_path)
        return pd.read_csv(io.BytesIO(raw))
    else:
        return pd.read_csv(storage_path)


def storage_mode() -> str:
    """Returns 'supabase' or 'local' — useful for health-check endpoint."""
    return "supabase" if is_cloud() else "local"
