"""pipeline_core/utils.py — Shared utilities (atomic write, JSON safety, etc.)."""

import os, json, hashlib


def atomic_write_json(data, path: str, schema_version: str = None):
    """Write JSON atomically via tmp + replace.

    Adds schema_version, generated_at, and source_sha256 metadata if schema_version provided.
    """
    if schema_version:
        if isinstance(data, dict):
            data = dict(data)  # shallow copy
            data["_meta"] = {
                "schema_version": schema_version,
                "generated_at": __import__("time").time(),
            }
        elif isinstance(data, list):
            data = {"_items": data, "_meta": {"schema_version": schema_version, "generated_at": __import__("time").time()}}

    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json_safe(path: str, default=None):
    """Read JSON, return default on any error."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def validate_schema_version(data, expected: str) -> bool:
    """Check if data has the expected schema version."""
    if isinstance(data, dict):
        meta = data.get("_meta", {})
        return meta.get("schema_version") == expected
    return False


def sha256_digest(obj) -> str:
    """Deterministic SHA256 of a JSON-serializable object."""
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def safe_filename(name: str, max_len: int = 100) -> str:
    """Sanitize a filename: keep alphanumeric, dash, dot, underscore, space."""
    safe = "".join(c for c in name if c.isalnum() or c in "._- ")[:max_len]
    return safe.strip() or "untitled"


def format_duration(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
