from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from urllib import error, parse, request

from trial_agent.models import ToolError


@dataclass(slots=True)
class ToolResult:
    records: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(slots=True)
class HTTPConfig:
    base_url: str
    timeout_s: float = 20.0
    api_key_env: str | None = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    default_headers: dict[str, str] = field(default_factory=lambda: {"Accept": "application/json"})


class ToolAdapter:
    name: str = "tool"

    def discover(self, query: str, filters: dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def fetch(self, trial_id: str) -> ToolResult:
        raise NotImplementedError

    def enrich(self, trial_record: dict[str, Any]) -> ToolResult:
        return ToolResult(records=[], metadata={"note": "enrichment not supported"})


class HTTPToolAdapter(ToolAdapter):
    """Small stdlib HTTP helper used by concrete adapters.

    This avoids adding runtime dependencies and keeps adapters easily testable.
    """

    def __init__(self, config: HTTPConfig) -> None:
        self.http = config

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(self.http.default_headers)
        if self.http.api_key_env:
            token = os.getenv(self.http.api_key_env)
            if token:
                headers[self.http.api_key_header] = f"{self.http.api_key_prefix}{token}"
        if extra:
            headers.update(extra)
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_404: bool = False,
    ) -> dict[str, Any]:
        url = self.http.base_url.rstrip("/") + "/" + path.lstrip("/")
        if params:
            encoded = parse.urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
            if encoded:
                url = f"{url}?{encoded}"

        payload: bytes | None = None
        req_headers = self._headers(headers)
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")

        req = request.Request(url, data=payload, method=method.upper(), headers=req_headers)
        try:
            with request.urlopen(req, timeout=self.http.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            if allow_404 and exc.code == 404:
                return {}
            message = exc.read().decode("utf-8", errors="ignore")
            raise ToolError(f"{self.name} HTTP {exc.code} for {path}: {message[:500]}") from exc
        except error.URLError as exc:
            raise ToolError(f"{self.name} network error for {path}: {exc}") from exc

        if not raw.strip():
            return {}
        try:
            parsed_json = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolError(f"{self.name} returned non-JSON response for {path}") from exc

        if not isinstance(parsed_json, dict):
            return {"data": parsed_json}
        return parsed_json

    def _request_text(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_404: bool = False,
    ) -> str:
        url = self.http.base_url.rstrip("/") + "/" + path.lstrip("/")
        if params:
            encoded = parse.urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
            if encoded:
                url = f"{url}?{encoded}"

        payload: bytes | None = None
        req_headers = self._headers(headers)
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")

        req = request.Request(url, data=payload, method=method.upper(), headers=req_headers)
        try:
            with request.urlopen(req, timeout=self.http.timeout_s) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except error.HTTPError as exc:
            if allow_404 and exc.code == 404:
                return ""
            message = exc.read().decode("utf-8", errors="ignore")
            raise ToolError(f"{self.name} HTTP {exc.code} for {path}: {message[:500]}") from exc
        except error.URLError as exc:
            raise ToolError(f"{self.name} network error for {path}: {exc}") from exc
