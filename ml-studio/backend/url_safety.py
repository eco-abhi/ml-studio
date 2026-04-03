"""Validate HTTP(S) URLs before server-side fetch to reduce SSRF risk."""

import ipaddress
from urllib.parse import urlparse

import requests
from fastapi import HTTPException


def _host_blocked(host: str) -> bool:
    h = (host or "").lower().strip("[]")
    if h in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    if h == "169.254.169.254" or h.startswith("169.254."):
        return True
    try:
        ip = ipaddress.ip_address(h)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    except ValueError:
        pass
    for suffix in (".local", ".internal", ".localhost"):
        if h.endswith(suffix):
            return True
    return False


def validate_public_http_url(url: str) -> str:
    """Ensure URL is http(s) with a host that is not obviously internal."""
    raw = (url or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="URL must use http or https")
    host = parsed.hostname
    if not host:
        raise HTTPException(status_code=400, detail="Invalid URL: missing host")
    if _host_blocked(host):
        raise HTTPException(status_code=400, detail="URL host is not allowed")
    return raw


def fetch_url_bytes(url: str, timeout: int = 30) -> tuple[bytes, str]:
    """GET a URL after validation. Redirects are disabled to avoid bypassing host checks."""
    safe = validate_public_http_url(url)
    resp = requests.get(safe, timeout=timeout, allow_redirects=False)
    resp.raise_for_status()
    path_last = urlparse(safe).path.rsplit("/", 1)[-1] or "data.csv"
    filename = path_last if path_last.endswith(".csv") else f"{path_last}.csv"
    return resp.content, filename
