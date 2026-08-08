"""tests/security/test_csp.py
==============================
Unit tests for the strict Content Security Policy (CSP) implementation
introduced in Issue #1561.

Covers:
* :func:`src.security.csp.build_csp_header` — directive presence, token
  blocking (eval, unsafe-inline in scripts), environment-variable driven
  allow-list extensions, dev-mode relaxation, report-uri injection.
* :class:`src.asgi_app.SecurityHeadersMiddleware` — verifies that the
  upgraded middleware injects all required security headers on every response.
"""

from __future__ import annotations

import os

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.security.csp import build_csp_header, _parse_env_hosts, _is_truthy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_csp(header: str) -> dict[str, list[str]]:
    """Parse a CSP header string into {directive: [token, ...]} dict."""
    result: dict[str, list[str]] = {}
    for part in header.split(";"):
        tokens = part.strip().split()
        if tokens:
            result[tokens[0]] = tokens[1:]
    return result


# ---------------------------------------------------------------------------
# build_csp_header – structural / baseline tests
# ---------------------------------------------------------------------------


class TestBuildCspHeaderBaseline:
    """Basic structure and required directives."""

    def test_returns_non_empty_string(self):
        policy = build_csp_header()
        assert isinstance(policy, str)
        assert len(policy) > 0

    def test_contains_default_src(self):
        policy = build_csp_header()
        assert "default-src" in policy

    def test_contains_script_src(self):
        policy = build_csp_header()
        assert "script-src" in policy

    def test_contains_style_src(self):
        policy = build_csp_header()
        assert "style-src" in policy

    def test_contains_img_src(self):
        policy = build_csp_header()
        assert "img-src" in policy

    def test_contains_connect_src(self):
        policy = build_csp_header()
        assert "connect-src" in policy

    def test_contains_frame_ancestors_none(self):
        policy = build_csp_header()
        directives = _parse_csp(policy)
        assert "frame-ancestors" in directives
        assert "'none'" in directives["frame-ancestors"]

    def test_contains_object_src_none(self):
        """object-src 'none' prevents <object>/<embed> plugin injection."""
        directives = _parse_csp(build_csp_header())
        assert "object-src" in directives
        assert "'none'" in directives["object-src"]

    def test_contains_base_uri_self(self):
        """base-uri 'self' prevents <base> tag injection attacks."""
        directives = _parse_csp(build_csp_header())
        assert "base-uri" in directives
        assert "'self'" in directives["base-uri"]

    def test_contains_form_action_self(self):
        directives = _parse_csp(build_csp_header())
        assert "form-action" in directives
        assert "'self'" in directives["form-action"]

    def test_directives_separated_by_semicolons(self):
        policy = build_csp_header()
        # Every part after split should start with a known directive keyword
        for part in policy.split(";"):
            tokens = part.strip().split()
            assert len(tokens) >= 1, f"Empty directive part in policy: {policy!r}"


# ---------------------------------------------------------------------------
# build_csp_header – XSS / injection blocking
# ---------------------------------------------------------------------------


class TestCspXssBlocking:
    """The strict policy must block eval() and arbitrary inline scripts."""

    def test_no_unsafe_eval_by_default(self):
        policy = build_csp_header()
        assert "'unsafe-eval'" not in policy, (
            "unsafe-eval must not be present in the production policy"
        )

    def test_no_unsafe_inline_in_script_src(self):
        """script-src must never contain 'unsafe-inline' (XSS vector)."""
        directives = _parse_csp(build_csp_header())
        assert "'unsafe-inline'" not in directives.get("script-src", []), (
            "unsafe-inline in script-src completely bypasses CSP script filtering"
        )

    def test_strict_dynamic_present_in_script_src(self):
        """'strict-dynamic' allows trusted script propagation via nonces/hashes."""
        directives = _parse_csp(build_csp_header())
        assert "'strict-dynamic'" in directives["script-src"]

    def test_self_present_in_default_src(self):
        directives = _parse_csp(build_csp_header())
        assert "'self'" in directives["default-src"]


# ---------------------------------------------------------------------------
# build_csp_header – trusted domain allow-lists
# ---------------------------------------------------------------------------


class TestCspAllowListedDomains:
    """Known trusted domains must appear in the correct directives."""

    def test_supabase_in_connect_src(self):
        directives = _parse_csp(build_csp_header())
        connect = " ".join(directives.get("connect-src", []))
        assert "supabase.co" in connect, "Supabase API must be in connect-src"

    def test_supabase_storage_in_img_src(self):
        directives = _parse_csp(build_csp_header())
        img = " ".join(directives.get("img-src", []))
        assert "supabase" in img, "Supabase storage must be in img-src"

    def test_plotly_cdn_in_script_src(self):
        directives = _parse_csp(build_csp_header())
        script = " ".join(directives.get("script-src", []))
        assert "cdn.plot.ly" in script, "Plotly CDN must be in script-src"

    def test_google_fonts_stylesheet_in_style_src(self):
        directives = _parse_csp(build_csp_header())
        style = " ".join(directives.get("style-src", []))
        assert "fonts.googleapis.com" in style

    def test_google_fonts_files_in_font_src(self):
        directives = _parse_csp(build_csp_header())
        fonts = " ".join(directives.get("font-src", []))
        assert "fonts.gstatic.com" in fonts

    def test_data_uri_in_img_src(self):
        """Streamlit renders base64 PNGs; data: must be allowed."""
        directives = _parse_csp(build_csp_header())
        assert "data:" in directives.get("img-src", [])

    def test_websocket_in_connect_src(self):
        """Streamlit uses WebSocket connections."""
        directives = _parse_csp(build_csp_header())
        connect_str = " ".join(directives.get("connect-src", []))
        assert "wss:" in connect_str or "ws:" in connect_str


# ---------------------------------------------------------------------------
# build_csp_header – environment-variable extensions
# ---------------------------------------------------------------------------


class TestCspEnvVarExtensions:
    """Extra allow-list entries injected via environment variables."""

    def test_extra_script_host_added_to_script_src(self, monkeypatch):
        monkeypatch.setenv("CSP_EXTRA_SCRIPT_HOSTS", "https://extra-cdn.example.com")
        policy = build_csp_header()
        assert "https://extra-cdn.example.com" in policy

    def test_extra_connect_host_added_to_connect_src(self, monkeypatch):
        monkeypatch.setenv("CSP_EXTRA_CONNECT_HOSTS", "https://myproject.supabase.co")
        policy = build_csp_header()
        assert "https://myproject.supabase.co" in policy

    def test_extra_img_host_added_to_img_src(self, monkeypatch):
        monkeypatch.setenv("CSP_EXTRA_IMG_HOSTS", "https://img.example.com")
        directives = _parse_csp(build_csp_header())
        assert "https://img.example.com" in directives["img-src"]

    def test_multiple_extra_hosts_comma_separated(self, monkeypatch):
        monkeypatch.setenv(
            "CSP_EXTRA_SCRIPT_HOSTS",
            "https://a.example.com, https://b.example.com",
        )
        policy = build_csp_header()
        assert "https://a.example.com" in policy
        assert "https://b.example.com" in policy

    def test_empty_env_var_adds_no_hosts(self, monkeypatch):
        monkeypatch.setenv("CSP_EXTRA_SCRIPT_HOSTS", "")
        policy = build_csp_header()
        directives = _parse_csp(policy)
        # Baseline tokens only — no empty string in script-src
        assert "" not in directives.get("script-src", [])

    def test_whitespace_only_env_var_ignored(self, monkeypatch):
        monkeypatch.setenv("CSP_EXTRA_SCRIPT_HOSTS", "   ,  , ")
        policy = build_csp_header()
        assert "  " not in policy


# ---------------------------------------------------------------------------
# build_csp_header – dev mode
# ---------------------------------------------------------------------------


class TestCspDevMode:
    """Development mode adds unsafe-eval for Vite HMR."""

    def test_dev_mode_adds_unsafe_eval_via_arg(self):
        policy = build_csp_header(dev_mode=True)
        assert "'unsafe-eval'" in policy

    def test_dev_mode_adds_unsafe_eval_via_env(self, monkeypatch):
        monkeypatch.setenv("ENABLE_CSP_DEV_MODE", "true")
        policy = build_csp_header()
        assert "'unsafe-eval'" in policy

    def test_dev_mode_false_no_unsafe_eval(self):
        policy = build_csp_header(dev_mode=False)
        assert "'unsafe-eval'" not in policy

    def test_dev_mode_env_false_string(self, monkeypatch):
        monkeypatch.setenv("ENABLE_CSP_DEV_MODE", "false")
        policy = build_csp_header()
        assert "'unsafe-eval'" not in policy

    def test_production_policy_has_no_unsafe_eval(self):
        """Regression guard: calling without args must never return unsafe-eval."""
        assert "'unsafe-eval'" not in build_csp_header()


# ---------------------------------------------------------------------------
# build_csp_header – report-uri
# ---------------------------------------------------------------------------


class TestCspReportUri:
    """report-uri directive is appended only when configured."""

    def test_report_uri_via_arg(self):
        policy = build_csp_header(report_uri="https://csp.example.com/report")
        assert "report-uri https://csp.example.com/report" in policy

    def test_report_uri_via_env(self, monkeypatch):
        monkeypatch.setenv("CSP_REPORT_URI", "https://csp.example.com/report")
        policy = build_csp_header()
        assert "report-uri" in policy
        assert "https://csp.example.com/report" in policy

    def test_no_report_uri_by_default(self, monkeypatch):
        monkeypatch.delenv("CSP_REPORT_URI", raising=False)
        policy = build_csp_header()
        assert "report-uri" not in policy

    def test_empty_report_uri_arg_suppresses_directive(self, monkeypatch):
        monkeypatch.setenv("CSP_REPORT_URI", "https://csp.example.com/report")
        policy = build_csp_header(report_uri="")
        assert "report-uri" not in policy


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestParseEnvHosts:
    """Unit tests for the internal _parse_env_hosts helper."""

    def test_single_host(self, monkeypatch):
        monkeypatch.setenv("_TEST_HOSTS", "https://a.example.com")
        assert _parse_env_hosts("_TEST_HOSTS") == ["https://a.example.com"]

    def test_multiple_hosts(self, monkeypatch):
        monkeypatch.setenv("_TEST_HOSTS", "https://a.example.com,https://b.example.com")
        result = _parse_env_hosts("_TEST_HOSTS")
        assert "https://a.example.com" in result
        assert "https://b.example.com" in result

    def test_empty_env_returns_empty_list(self, monkeypatch):
        monkeypatch.setenv("_TEST_HOSTS", "")
        assert _parse_env_hosts("_TEST_HOSTS") == []

    def test_missing_env_returns_empty_list(self, monkeypatch):
        monkeypatch.delenv("_TEST_HOSTS", raising=False)
        assert _parse_env_hosts("_TEST_HOSTS") == []

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("_TEST_HOSTS", " https://a.example.com , https://b.example.com ")
        result = _parse_env_hosts("_TEST_HOSTS")
        assert "https://a.example.com" in result
        assert "https://b.example.com" in result


class TestIsTruthy:
    """Unit tests for the internal _is_truthy helper."""

    @pytest.mark.parametrize("val", ["true", "1", "yes", "on", "TRUE", "YES", "ON"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("_TEST_FLAG", val)
        assert _is_truthy("_TEST_FLAG") is True

    @pytest.mark.parametrize("val", ["false", "0", "no", "off", "", "False"])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("_TEST_FLAG", val)
        assert _is_truthy("_TEST_FLAG") is False

    def test_missing_env_is_falsy(self, monkeypatch):
        monkeypatch.delenv("_TEST_FLAG", raising=False)
        assert _is_truthy("_TEST_FLAG") is False


# ---------------------------------------------------------------------------
# SecurityHeadersMiddleware integration tests
# ---------------------------------------------------------------------------


async def _echo_handler(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


def _make_test_client() -> TestClient:
    """Build a minimal Starlette app with SecurityHeadersMiddleware attached."""
    from src.asgi_app import SecurityHeadersMiddleware

    test_app = Starlette(
        routes=[Route("/ping", _echo_handler, methods=["GET"])],
        middleware=[Middleware(SecurityHeadersMiddleware)],
    )
    return TestClient(test_app, raise_server_exceptions=False)


class TestSecurityHeadersMiddleware:
    """Verify that SecurityHeadersMiddleware injects all required headers."""

    @pytest.fixture(autouse=True)
    def client(self):
        self._client = _make_test_client()

    def _response(self):
        return self._client.get("/ping")

    def test_csp_header_present(self):
        resp = self._response()
        assert "content-security-policy" in resp.headers

    def test_csp_header_contains_default_src(self):
        resp = self._response()
        assert "default-src" in resp.headers["content-security-policy"]

    def test_csp_header_blocks_unsafe_eval(self):
        resp = self._response()
        assert "'unsafe-eval'" not in resp.headers["content-security-policy"]

    def test_csp_header_blocks_frame_embedding(self):
        resp = self._response()
        csp = resp.headers["content-security-policy"]
        assert "frame-ancestors" in csp
        assert "'none'" in csp

    def test_x_frame_options_deny(self):
        resp = self._response()
        assert resp.headers.get("x-frame-options") == "DENY"

    def test_x_content_type_options_nosniff(self):
        resp = self._response()
        assert resp.headers.get("x-content-type-options") == "nosniff"

    def test_x_xss_protection(self):
        resp = self._response()
        assert resp.headers.get("x-xss-protection") == "1; mode=block"

    def test_referrer_policy(self):
        resp = self._response()
        assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"

    def test_permissions_policy_present(self):
        resp = self._response()
        assert "permissions-policy" in resp.headers

    def test_permissions_policy_blocks_camera(self):
        resp = self._response()
        assert "camera=()" in resp.headers.get("permissions-policy", "")

    def test_permissions_policy_blocks_microphone(self):
        resp = self._response()
        assert "microphone=()" in resp.headers.get("permissions-policy", "")

    def test_no_hsts_without_env(self, monkeypatch):
        monkeypatch.delenv("ENABLE_HSTS", raising=False)
        resp = self._response()
        assert "strict-transport-security" not in resp.headers

    def test_hsts_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("ENABLE_HSTS", "true")
        # Rebuild client so middleware picks up new env var
        client = _make_test_client()
        resp = client.get("/ping")
        hsts = resp.headers.get("strict-transport-security", "")
        assert "max-age=" in hsts
        assert "includeSubDomains" in hsts

    def test_headers_present_on_every_response(self):
        """All security headers are attached regardless of endpoint."""
        for _ in range(3):
            resp = self._response()
            assert "content-security-policy" in resp.headers
            assert "x-frame-options" in resp.headers
            assert "x-content-type-options" in resp.headers
