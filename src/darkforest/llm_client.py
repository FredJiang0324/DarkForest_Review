from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - exercised only when requests is absent
    requests = None

from .schemas import LLMResponse
from .utils import normalize_usage


class VLLMClient:
    def __init__(
        self,
        endpoint: str,
        model_name: str,
        timeout_sec: float = 120.0,
        max_retries: int = 3,
        api_style: str = "completions",
    ) -> None:
        self.endpoint = self._resolve_endpoint(endpoint, api_style)
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.api_style = api_style

    @staticmethod
    def _resolve_endpoint(endpoint: str, api_style: str) -> str:
        if api_style == "chat" and endpoint.endswith("/v1/completions"):
            return endpoint[: -len("/v1/completions")] + "/v1/chat/completions"
        if api_style == "completions" and endpoint.endswith("/v1/chat/completions"):
            return endpoint[: -len("/v1/chat/completions")] + "/v1/completions"
        return endpoint

    def complete(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        if self.api_style == "chat":
            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        else:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        if seed is not None:
            payload["seed"] = seed

        last_error: Optional[str] = None
        start = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                response_json = self._post_json(payload)
                latency = time.perf_counter() - start
                text = extract_completion_text(response_json)
                usage = normalize_usage(response_json.get("usage"), prompt, text)
                return LLMResponse(
                    text=text,
                    latency_sec=latency,
                    usage=usage,
                    error=None,
                    raw_json=response_json,
                )
            except Exception as exc:  # noqa: BLE001 - all model call errors are recorded
                last_error = str(exc)
                if attempt >= self.max_retries:
                    break
                time.sleep(min(8.0, 0.5 * (2**attempt)))

        latency = time.perf_counter() - start
        usage = normalize_usage(None, prompt, "")
        return LLMResponse(text="", latency_sec=latency, usage=usage, error=last_error)

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if requests is not None:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            return response.json()

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def extract_completion_text(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    choice = choices[0] or {}
    if "text" in choice and choice["text"] is not None:
        return str(choice["text"])
    message = choice.get("message")
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"])
    return ""
