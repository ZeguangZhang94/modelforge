#!/usr/bin/env python3
"""Mock 推理引擎：模拟 llama-server 的最小接口，用于无 GPU 环境下的全链路测试。

支持：--port --model --ctx-size --n-gpu-layers --parallel --metrics --version
接口：/health、/v1/models、/v1/chat/completions（非流式 + SSE 流式）
"""

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": "mock-model", "object": "model", "created": 0, "owned_by": "mock"}]})
        if self.path == "/metrics":
            return self._json(200, {"prompt_tokens_total": 0})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": "not found"})
        body = json.loads(self.rfile.read(
            int(self.headers.get("Content-Length", 0))).decode("utf-8", "replace"))
        user = body.get("messages", [{}])[-1].get("content", "")
        reply = f"[mock-engine] 收到：{user}（n_gpu_layers={ARGS.get('ngl')}, ctx={ARGS.get('ctx')}）"
        prompt_tok = sum(len(m.get("content", "")) for m in body.get("messages", [])) // 4
        usage = {"prompt_tokens": prompt_tok, "completion_tokens": len(reply) // 4,
                 "total_tokens": prompt_tok + len(reply) // 4}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in [reply[i:i + 8] for i in range(0, len(reply), 8)]:
                data = {"id": "mock", "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": chunk},
                                     "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.02)
            # llama.cpp 风格：include_usage 时最后发一个仅含 usage 的块
            if body.get("stream_options", {}).get("include_usage"):
                final = {"id": "mock", "object": "chat.completion.chunk",
                         "choices": [], "usage": usage,
                         "timings": {"predicted_per_second": 42.0}}
                self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        return self._json(200, {
            "id": "mock", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": reply}}],
            "usage": usage})


def main():
    if "--version" in sys.argv:
        print("mock-llama-server version e2e-1.0")
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--model", default="")
    ap.add_argument("--ctx-size", type=int, default=8192)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    a, _ = ap.parse_known_args()
    ARGS.update({"ngl": a.n_gpu_layers, "ctx": a.ctx_size})
    print(f"mock-engine listening on {a.host}:{a.port} "
          f"(model={a.model}, ngl={a.n_gpu_layers}, ctx={a.ctx_size})")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
