#!/usr/bin/env python3
"""Preloaded Laya judge server for the local decision backend.

Keeps the model warm so every agent decision is one ~25 ms HTTP round trip
to 127.0.0.1 instead of a fresh model load. Run once, leave running:

    uv run --extra laya python scripts/layad.py

Endpoints:
  GET  /healthz
  POST /judge   {"state": ..., "questions": ...} -> raw laya answers
"""

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("USE_TF", "0")  # model card: avoids TF/abseil hang on load

import laya  # noqa: E402

AGENT = None
LOCK = threading.Lock()  # MPS predict is not concurrency-safe


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            self._send(200, {"ok": True, "device": str(AGENT.device)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("content-length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad request: {e}"})
        if self.path != "/judge":
            return self._send(404, {"error": "not found"})
        try:
            t0 = time.time()
            with LOCK:
                res = AGENT.predict(req.get("state", {}), req.get("questions", {}))
            res["ms"] = round((time.time() - t0) * 1000, 1)
            self._send(200, res)
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *args):  # keep stdout to the ready line
        pass


def main():
    global AGENT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    t0 = time.time()
    AGENT = laya.load(args.model, device=args.device)
    AGENT.predict(  # warmup: first real call stays ~25ms
        {"text": "warmup"},
        {"ok": {"type": "choice", "instructions": "Is this a warmup?",
                "criteria": {"yes": "warmup text", "no": "real task"}}},
    )
    print(f"layad ready on 127.0.0.1:{args.port} device={AGENT.device} "
          f"load={time.time() - t0:.1f}s", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
