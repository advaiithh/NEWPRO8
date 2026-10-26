"""Dashboard: stdlib HTTP + Server-Sent Events on loopback. Updates at 2 Hz so it does not distort measurements."""
from __future__ import annotations

import collections
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from pv.config import config, path
from pv.guard import gpucheck, netguard

HERE = Path(__file__).resolve().parent


class State:
    def __init__(self, pipe, rt, gov) -> None:
        self.pipe, self.rt, self.gov = pipe, rt, gov
        self.lat: collections.deque[float] = collections.deque(maxlen=50)
        self.last: dict[str, Any] = {}
        self.turns = 0
        self._proof: dict[str, Any] = {}
        self._proof_t = 0.0
        prev = pipe.on_turn

        def on_turn(tl) -> None:
            self.turns += 1
            if tl.latency_ms is not None:
                self.lat.append(tl.latency_ms)
            self.last = {"latency_ms": tl.latency_ms, "stages_ms": tl.stages_ms(), "path": tl.path, "tier": tl.tier,
                         "sim": tl.similarity, "hangover_ms": tl.hangover_ms, "transcript": tl.transcript,
                         "reply": tl.reply, "intent": tl.extra.get("intent") or tl.extra.get("skill") or ""}
            if prev:
                prev(tl)
        pipe.on_turn = on_turn

    def proof(self) -> dict[str, Any]:
        if time.time() - self._proof_t > 5:
            try:
                self._proof = gpucheck.proof(config["ollama"]["url"])
            except Exception as e:          # Ollama restarting must not kill the dashboard
                self._proof = {"error": repr(e), "pass": False}
            self._proof_t = time.time()
        return self._proof

    def snapshot(self) -> dict[str, Any]:
        lim, s = self.rt.limiter, self.pipe.tier
        lat = list(self.lat)
        report = path("reports", "summary.json")
        return {
            "limit": {"cores": lim.cores, "ram_mb": lim.ram_mb, "banner": lim.banner()},
            "rss_mb": lim.rss_mb(), "cpu_pct": self.rt.sampler.cpu_percent,
            "tok_s": self.pipe.llm.stats.tokens_per_s, "watts": self.rt.energy.watts() if self.rt.energy.available else None,
            "tier": {"id": s.id, "name": s.name, "forced": bool(self.gov and self.gov.forced)},
            "transitions": (self.gov.log[-8:] if self.gov else []),
            "last": self.last, "turns": self.turns,
            "p50": float(np.percentile(lat, 50)) if lat else None, "p95": float(np.percentile(lat, 95)) if lat else None,
            "proof": self.proof(), "net": netguard.status(),
            "models_local": True,
            "report": json.loads(report.read_text(encoding="utf-8")) if report.exists() else None,
        }


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while True:
                        self.wfile.write(f"data: {json.dumps(state.snapshot(), default=str)}\n\n".encode())
                        self.wfile.flush()
                        time.sleep(0.5)
                except (BrokenPipeError, ConnectionError, OSError):
                    return
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/tighten":
                state.rt.limiter.tighten(int(body["cores"]), int(body["ram_mb"]))
            elif self.path == "/api/tier":
                if body["tier"] == "auto":
                    state.gov and state.gov.release_force()
                elif state.gov:
                    state.gov.force_tier(int(body["tier"]), "dashboard control")
                else:
                    state.pipe.set_tier(int(body["tier"]))
            else:
                return self._send(404, b"not found", "text/plain")
            self._send(200, b"{}", "application/json")
    return H


def start(pipe, rt, gov) -> ThreadingHTTPServer:
    state = State(pipe, rt, gov)
    srv = ThreadingHTTPServer((config["dash"]["host"], config["dash"]["port"]), make_handler(state))
    threading.Thread(target=srv.serve_forever, name="dash", daemon=True).start()
    print(f"[dashboard] http://{config['dash']['host']}:{config['dash']['port']}")
    return srv
