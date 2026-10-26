"""Streaming Ollama client over loopback. CPU only (num_gpu=0), cancellable for barge-in."""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from pv.config import config


@dataclass
class LLMStats:
    first_token_s: float | None = None
    tokens: int = 0
    gen_s: float = 0.0
    prompt_tokens: int = 0
    load_s: float = 0.0
    cancelled: bool = False

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.gen_s if self.gen_s > 0 else 0.0


class LLMBackend(ABC):
    @abstractmethod
    def stream(self, messages: list[dict[str, str]], **opts: Any) -> Iterator[str]: ...

    @abstractmethod
    def cancel(self) -> None: ...


@dataclass
class OllamaOptions:
    model: str
    num_ctx: int | None = 1024
    num_predict: int | None = 80
    threads: int | None = None
    default_options: bool = False         # baseline: send only the model, nothing else
    think: bool | None = None             # False for reasoning models (Qwen3.x): thinking tokens wreck voice latency

    def api_options(self) -> dict[str, Any]:
        if self.default_options:
            return {"num_gpu": 0}         # CPU-only is a hard rule even for the baseline
        o: dict[str, Any] = {"num_gpu": 0, "temperature": config["ollama"]["temperature"],
                             "repeat_penalty": config["ollama"]["repeat_penalty"]}
        if self.num_ctx:
            o["num_ctx"] = self.num_ctx
        if self.num_predict:
            o["num_predict"] = self.num_predict
        if self.threads:
            o["num_thread"] = self.threads
        return o


class OllamaClient(LLMBackend):
    def __init__(self, url: str | None = None) -> None:
        self.url = url or config["ollama"]["url"]
        self.http = httpx.Client(base_url=self.url, timeout=httpx.Timeout(300.0, connect=5.0))
        self.stats = LLMStats()
        self._resp: httpx.Response | None = None
        self._cancel = threading.Event()

    def _body(self, messages: list[dict[str, str]], o: OllamaOptions, stream: bool = True) -> dict[str, Any]:
        body = {"model": o.model, "messages": messages, "stream": stream, "keep_alive": -1, "options": o.api_options()}
        if o.think is not None:
            body["think"] = o.think
        return body

    def stream(self, messages: list[dict[str, str]], **opts: Any) -> Iterator[str]:
        o: OllamaOptions = opts["options"]
        self.stats = LLMStats()
        self._cancel.clear()
        t0 = time.perf_counter()
        t_first: float | None = None
        with self.http.stream("POST", "/api/chat", json=self._body(messages, o)) as resp:
            self._resp = resp
            resp.raise_for_status()
            try:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("done"):
                        self.stats.tokens = d.get("eval_count", self.stats.tokens)
                        self.stats.gen_s = d.get("eval_duration", 0) / 1e9
                        self.stats.prompt_tokens = d.get("prompt_eval_count", 0)
                        self.stats.load_s = d.get("load_duration", 0) / 1e9
                        break
                    tok = d.get("message", {}).get("content", "")
                    if tok:
                        if t_first is None:
                            t_first = time.perf_counter()
                            self.stats.first_token_s = t_first - t0
                        yield tok
            except (httpx.ReadError, httpx.StreamClosed, httpx.RemoteProtocolError):
                if not self._cancel.is_set():
                    raise
                self.stats.cancelled = True
            finally:
                self._resp = None

    def cancel(self) -> None:
        """Barge-in: close the stream now; the server stops generating."""
        self._cancel.set()
        resp = self._resp
        if resp is not None:
            resp.close()

    def warm(self, messages: list[dict[str, str]], o: OllamaOptions) -> float:
        """D3 prefix warming: one-token request so system prompt + history sit in the KV cache."""
        t0 = time.perf_counter()
        warm = OllamaOptions(o.model, o.num_ctx, 1, o.threads, o.default_options, o.think)
        body = self._body(messages, warm, stream=False)
        self.http.post("/api/chat", json=body).raise_for_status()
        return time.perf_counter() - t0

    def unload(self, model: str) -> None:
        self.http.post("/api/chat", json={"model": model, "messages": [], "keep_alive": 0}).raise_for_status()

    def ps(self) -> list[dict[str, Any]]:
        return self.http.get("/api/ps").json().get("models", [])

    def close(self) -> None:
        self.http.close()
