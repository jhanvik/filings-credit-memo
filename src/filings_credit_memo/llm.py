"""A minimal LLM interface, a local Ollama implementation and an on-disk cache/log.

Everything in the pipeline talks to the `LLM` protocol, so a paid API can be swapped in later without touching the
extraction code. The Ollama request format below (POST /api/chat with a JSON-schema `format`) is from the author's
knowledge of Ollama's structured-output feature: verify it against the Ollama docs if a call fails.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

import httpx

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"


class LLMError(RuntimeError):
    """The model could not be reached or answered with an error."""


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, schema: dict[str, Any] | None = None) -> str: ...


class OllamaLLM:
    """Local model through Ollama. Temperature 0 and a fixed seed so reruns are as repeatable as the model allows."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_OLLAMA_URL,
        timeout: float = 600.0,
        num_ctx: int = 8192,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = f"ollama:{model}"
        self._model = model
        self._num_ctx = num_ctx
        self._http = httpx.Client(base_url=host, timeout=timeout, transport=transport)

    def complete(self, system: str, user: str, schema: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0, "seed": 42, "num_ctx": self._num_ctx},
        }
        if schema is not None:
            payload["format"] = schema
        try:
            response = self._http.post("/api/chat", json=payload)
        except httpx.TransportError as err:
            raise LLMError(f"cannot reach Ollama ({err}). Is it running? Try: brew services start ollama") from err
        if response.status_code != 200:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            return str(response.json()["message"]["content"])
        except (ValueError, KeyError, TypeError) as err:
            raise LLMError("unexpected response format from Ollama") from err

    def close(self) -> None:
        self._http.close()


class CachedLLM:
    """Wraps any LLM: identical requests are answered from disk, and every call is stored with its prompt (an audit log)."""

    def __init__(self, inner: LLM, cache_dir: Path) -> None:
        self.name = inner.name
        self._inner = inner
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0  # real model calls made through this wrapper

    def _path(self, system: str, user: str, schema: dict[str, Any] | None) -> Path:
        key = json.dumps([self.name, system, user, schema], sort_keys=True)
        return self._dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def complete(self, system: str, user: str, schema: dict[str, Any] | None = None) -> str:
        path = self._path(system, user, schema)
        if path.exists():
            return str(json.loads(path.read_text())["response"])
        response = self._inner.complete(system, user, schema)
        self.calls += 1
        record = {"model": self.name, "system": system, "user": user, "schema": schema, "response": response}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        os.replace(tmp, path)
        return response
