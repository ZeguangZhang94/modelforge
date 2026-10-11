#!/usr/bin/env python3
"""ModelForge MCP 服务器（JSON-RPC 2.0 over stdio，MCP 协议）。

让 AI 编程助手（Claude Code、Cursor 等）直接管理本地模型服务：
查看状态、查指标、重启引擎、直接对话。

Claude Desktop / Claude Code 配置示例：
{
  "mcpServers": {
    "modelforge": {
      "command": "python",
      "args": ["<安装包目录>/mcp_server.py",
               "--config", "<安装目录>/state/config.json"]
    }
  }
}

协议实现：MCP 2025 版 stdio 传输（每行一个 JSON-RPC 消息）。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

GATEWAY_PORT_FALLBACK = 8080


class McpServer:
    def __init__(self, config_path: Path) -> None:
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        # 优先读实际运行端口（可能自动顺延过）
        runtime = config_path.parent / "runtime.json"
        port = self.config.get("port", GATEWAY_PORT_FALLBACK)
        if runtime.exists():
            try:
                port = json.loads(runtime.read_text()).get("port", port)
            except Exception:
                pass
        self.base = f"http://127.0.0.1:{port}"

    # ---- MCP 方法 ----

    def tools(self) -> list[dict]:
        return [
            {"name": "model_status", "description": "查看本地模型服务状态（引擎是否在线、运行时长）",
             "inputSchema": {"type": "object", "properties": {}},
             "annotations": {"readOnlyHint": True}},
            {"name": "model_metrics", "description": "查看吞吐指标（请求数、token 统计）",
             "inputSchema": {"type": "object", "properties": {}},
             "annotations": {"readOnlyHint": True}},
            {"name": "model_info", "description": "查看模型信息（名称、档位、显卡、大小）",
             "inputSchema": {"type": "object", "properties": {}},
             "annotations": {"readOnlyHint": True}},
            {"name": "model_chat", "description": "与本地模型对话",
             "inputSchema": {"type": "object", "required": ["message"],
                             "properties": {"message": {"type": "string", "description": "用户消息"},
                                            "max_tokens": {"type": "integer", "default": 512}}}},
            {"name": "model_restart", "description": "重启推理引擎（模型卡住/无响应时使用）",
             "inputSchema": {"type": "object", "properties": {}}},
        ]

    def call(self, name: str, args: dict) -> str:
        if name == "model_info":
            b = dict(self.config.get("branding", {}))
            return json.dumps({
                "名称": b.get("display_name") or self.config.get("name"),
                "档位": self.config.get("variant"),
                "显卡": self.config.get("gpu"),
                "引擎": self.config.get("engine", {}).get("backend"),
            }, ensure_ascii=False)
        if name == "model_status":
            return self._get("/health")
        if name == "model_metrics":
            return self._get("/metrics")
        if name == "model_restart":
            req = urllib.request.Request(self.base + "/admin/restart", data=b"", method="POST")
            return self._open(req)
        if name == "model_chat":
            body = json.dumps({
                "model": "local", "stream": False,
                "max_tokens": int(args.get("max_tokens", 512)),
                "messages": [{"role": "user", "content": args.get("message", "")}],
            }).encode()
            req = urllib.request.Request(self.base + "/v1/chat/completions",
                                         data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            resp = json.loads(self._open(req))
            if "choices" in resp:
                return resp["choices"][0]["message"]["content"]
            return json.dumps(resp, ensure_ascii=False)
        raise ValueError(f"未知工具: {name}")

    # ---- HTTP ----

    def _open(self, req) -> str:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read().decode("utf-8", "replace")

    def _get(self, path: str) -> str:
        return self._open(urllib.request.Request(self.base + path))

    # ---- stdio 主循环 ----

    def serve(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = msg.get("id")
            method = msg.get("method", "")
            params = msg.get("params", {})
            try:
                if method == "initialize":
                    result = {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                              "capabilities": {"tools": {}},
                              "serverInfo": {"name": "modelforge", "version": "1.0"}}
                elif method == "tools/list":
                    result = {"tools": self.tools()}
                elif method == "tools/call":
                    text = self.call(params.get("name", ""), params.get("arguments", {}))
                    result = {"content": [{"type": "text", "text": text}]}
                elif method in ("notifications/initialized", "notifications/cancelled"):
                    continue
                else:
                    result = {}
                if mid is not None:
                    self._send({"jsonrpc": "2.0", "id": mid, "result": result})
            except Exception as e:
                if mid is not None:
                    self._send({"jsonrpc": "2.0", "id": mid,
                                "error": {"code": -32000, "message": str(e)}})

    @staticmethod
    def _send(obj: dict) -> None:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main() -> int:
    # MCP 走 stdio，强制 UTF-8（Windows 默认 GBK 会破坏 JSON 中的中文）
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    McpServer(args.config).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
