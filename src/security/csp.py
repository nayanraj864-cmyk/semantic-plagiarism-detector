"""src/security/csp.py
===================
Strict Content Security Policy (CSP) builder for the Semantic Plagiarism
Detector application.

This module constructs a production-grade, environment-aware CSP header value
that:

* Blocks all inline ``<script>`` execution and ``eval()`` calls (XSS defence).
* Allows only trusted, explicitly allow-listed domain sources for each fetch
  directive (script, style, image, font, connect, frame-ancestors, etc.).
* Supports optional development-mode relaxations controlled via environment
  variables so that the Vite / Streamlit HMR dev-server keeps working without
  disabling the policy entirely.
* Exposes a ``build_csp_header()`` helper so any WSGI/ASGI middleware can
  retrieve the full header value without duplicating configuration logic.

Accepted environment variables
-------------------------------
CSP_REPORT_URI
    If set, a ``report-uri`` directive is appended, allowing the browser to
    POST CSP violation reports to an endpoint of your choosing.  This is the
    standard mechanism for monitoring policy violations in production without
    blocking legitimate traffic.  Example::

        CSP_REPORT_URI=https://your-domain.com/csp-report

CSP_EXTRA_SCRIPT_HOSTS
    Comma-separated list of additional ``script-src`` hosts to allow-list.
    Useful for whitelisting a private CDN or a testing origin without editing
    the source.  Example::

        CSP_EXTRA_SCRIPT_HOSTS=https://cdn.your-domain.com

CSP_EXTRA_CONNECT_HOSTS
    Comma-separated list of additional ``connect-src`` hosts (e.g. Supabase
    project URLs that differ across deployment environments).  Example::

        CSP_EXTRA_CONNECT_HOSTS=https://abc123.supabase.co

CSP_EXTRA_IMG_HOSTS
    Comma-separated list of additional ``img-src`` hosts (object-storage or
    image CDN origins).  Example::

        CSP_EXTRA_IMG_HOSTS=https://images.your-cdn.com

ENABLE_CSP_DEV_MODE
    Set to ``true`` / ``1`` / ``yes`` to unlock ``unsafe-eval`` in
    ``script-src`` so that Vite HMR / browser devtools work.  **Never enable
    in production.**

References
----------
* https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Security-Policy
* https://csp.withgoogle.com/docs/strict-csp.html
* https://cheatsheetseries.owasp.org/cheatsheets/Content_Security_Policy_Cheat_Sheet.html
"""

from __future__ import annotations

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Allow-listed origins
# ---------------------------------------------------------------------------

# Supabase is used for authentication (connect-src) and optionally for
# avatar / storage bucket images (img-src).
_SUPABASE_API = "https://*.supabase.co"
_SUPABASE_STORAGE = "https://*.supabase.in"

# Plotly/CDN for optional chart bundles loaded at runtime.
_PLOTLY_CDN = "https://cdn.plot.ly"

# Google Fonts – stylesheet and font files.
_GFONTS_STYLE = "https://fonts.googleapis.com"
_GFONTS_FONT = "https://fonts.gstatic.com"

# Streamlit uses a small set of trusted CDN assets.
_STREAMLIT_ASSETS = "https://streamlit.io"

# Generic data-URI allow for small inline images (Streamlit renders base64 PNGs).
_DATA_URI = "data:"

# Blob URIs are used by some file-download helper widgets.
_BLOB_URI = "blob:"

# ---------------------------------------------------------------------------
# Policy directives
# ---------------------------------------------------------------------------

#: Base directive map that forms the strict production policy.
#: Values are *lists* of source tokens so callers can mutate them cleanly.
_BASE_DIRECTIVES: dict[str, list[str]] = {
    # ── Script ──────────────────────────────────────────────────────────────
    # 'strict-dynamic' combined with a nonce (or hash) allows trusted script
    # propagation while blocking unknown inline scripts.  We intentionally
    # omit 'unsafe-inline' to prevent XSS injection via injected <script> tags.
    # 'unsafe-eval' is NEVER included by default – see CSP_EXTRA_SCRIPT_HOSTS.
    "script-src": [
        "'self'",
        "'strict-dynamic'",
        _PLOTLY_CDN,
        _STREAMLIT_ASSETS,
    ],
    # ── Style ────────────────────────────────────────────────────────────────
    # Streamlit and Plotly inject style tags at runtime; we must allow
    # 'unsafe-inline' only for styles (much lower XSS risk than scripts).
    "style-src": [
        "'self'",
        "'unsafe-inline'",
        _GFONTS_STYLE,
    ],
    # ── Images ──────────────────────────────────────────────────────────────
    "img-src": [
        "'self'",
        _DATA_URI,
        _BLOB_URI,
        _SUPABASE_STORAGE,
    ],
    # ── Fonts ───────────────────────────────────────────────────────────────
    "font-src": [
        "'self'",
        _DATA_URI,
        _GFONTS_FONT,
    ],
    # ── XHR / Fetch / WebSocket ──────────────────────────────────────────────
    # Supabase REST and Auth endpoints + the Streamlit WebSocket endpoint.
    "connect-src": [
        "'self'",
        "wss:",
        "ws:",
        _SUPABASE_API,
        _SUPABASE_STORAGE,
    ],
    # ── Workers / Service Workers ────────────────────────────────────────────
    "worker-src": [
        "'self'",
        _BLOB_URI,
    ],
    # ── Frames ──────────────────────────────────────────────────────────────
    # Deny embedding in any frame to prevent clickjacking.
    "frame-ancestors": ["'none'"],
    # ── Default fallback ─────────────────────────────────────────────────────
    "default-src": ["'self'"],
    # ── Object / Embed ───────────────────────────────────────────────────────
    "object-src": ["'none'"],
    # ── Base URI ─────────────────────────────────────────────────────────────
    # Prevents <base> tag injection attacks.
    "base-uri": ["'self'"],
    # ── Form action ──────────────────────────────────────────────────────────
    "form-action": ["'self'"],
    # ── Manifest ─────────────────────────────────────────────────────────────
    "manifest-src": ["'self'"],
    # ── Media ────────────────────────────────────────────────────────────────
    "media-src": ["'self'", _DATA_URI, _BLOB_URI],
}


def _parse_env_hosts(env_var: str) -> list[str]:
    """Parse a comma-separated environment variable into a list of stripped hosts.

    Empty strings and whitespace-only tokens are silently discarded.

    Args:
        env_var: Name of the environment variable to read.

    Returns:
        List of non-empty host strings, or an empty list.
    """
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return []
    return [h.strip() for h in raw.split(",") if h.strip()]


def _is_truthy(env_var: str) -> bool:
    """Return True when an environment variable is set to a truthy string."""
    return os.getenv(env_var, "").strip().lower() in ("true", "1", "yes", "on")


def build_csp_header(
    report_uri: Optional[str] = None,
    dev_mode: Optional[bool] = None,
) -> str:
    """Build and return the full ``Content-Security-Policy`` header value.

    The returned string is ready to be assigned directly to the
    ``Content-Security-Policy`` HTTP response header.

    Args:
        report_uri:
            Override the ``CSP_REPORT_URI`` environment variable.  Set to an
            empty string to suppress the directive even if the env var is set.
        dev_mode:
            Override the ``ENABLE_CSP_DEV_MODE`` environment variable.  When
            ``True``, ``unsafe-eval`` is added to ``script-src`` to support
            Vite HMR.  **Never pass ``True`` in production code.**

    Returns:
        The full CSP policy string, e.g.::

            "default-src 'self'; script-src 'self' 'strict-dynamic' ...; ..."

    Examples:
        >>> policy = build_csp_header()
        >>> "default-src" in policy
        True
        >>> "unsafe-eval" not in policy  # never present by default
        True
        >>> policy_dev = build_csp_header(dev_mode=True)
        >>> "unsafe-eval" in policy_dev
        True
    """
    # Deep-copy the base directives so the module-level dict stays pristine.
    directives: dict[str, list[str]] = {
        key: list(values) for key, values in _BASE_DIRECTIVES.items()
    }

    # ── Environment-driven extra allow-list entries ───────────────────────────
    for host in _parse_env_hosts("CSP_EXTRA_SCRIPT_HOSTS"):
        if host not in directives["script-src"]:
            directives["script-src"].append(host)

    for host in _parse_env_hosts("CSP_EXTRA_CONNECT_HOSTS"):
        if host not in directives["connect-src"]:
            directives["connect-src"].append(host)

    for host in _parse_env_hosts("CSP_EXTRA_IMG_HOSTS"):
        if host not in directives["img-src"]:
            directives["img-src"].append(host)

    # ── Development-mode relaxation (unsafe-eval) ────────────────────────────
    _dev = dev_mode if dev_mode is not None else _is_truthy("ENABLE_CSP_DEV_MODE")
    if _dev and "'unsafe-eval'" not in directives["script-src"]:
        directives["script-src"].append("'unsafe-eval'")

    # ── Assemble directive strings ───────────────────────────────────────────
    parts: list[str] = []
    for directive, sources in directives.items():
        if sources:
            parts.append(f"{directive} {' '.join(sources)}")

    # ── Optional violation reporting ─────────────────────────────────────────
    _report_uri = report_uri if report_uri is not None else os.getenv("CSP_REPORT_URI", "")
    if _report_uri:
        parts.append(f"report-uri {_report_uri}")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Convenience constant: the production header value evaluated at import time.
# Re-evaluate dynamically per-request when env vars may change at runtime.
# ---------------------------------------------------------------------------
PRODUCTION_CSP: str = build_csp_header()
