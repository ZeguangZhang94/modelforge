#!/usr/bin/env python3
"""agentd —— 本地 AI API 网关 + 服务层（M2）。

职责（对应 docs/02-架构设计.md 第 3.5 节）：
- OpenAI 兼容代理：/v1/chat/completions、/v1/models、/v1/embeddings（含 SSE 流式透传）
- Anthropic 兼容适配：/v1/messages（协议转换到引擎）
- /health、/metrics（请求计数、token 吞吐、引擎在线状态）
- Web 控制台：/ （聊天 + 实时监控）
- 引擎看门狗：崩溃自动重启
- 安全：绑定非 loopback 地址时强制 API Key

用法：python serve.py --config <state/config.json> [--host x] [--port x]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
START_TIME = time.time()
DEFAULT_ENGINE_BASE = "http://127.0.0.1:18091"

STATS = {"requests_total": 0, "prompt_tokens": 0, "completion_tokens": 0,
         "errors_total": 0, "last_tps": 0.0}
STATS_LOCK = threading.Lock()

# 按密钥核算用量：key_name → {"requests", "tokens"}
USAGE_BY_KEY: dict[str, dict] = {}
AUDIT_LOCK = threading.Lock()

ROLES = ("consumer", "operator", "admin", "auditor")
# 路由 → 允许的角色（consumer 只能推理；operator 加运维；admin 全权；auditor 只读审计）
ROUTE_ROLES = {
    "view": ("consumer", "operator", "admin", "auditor"),
    "inference": ("consumer", "operator", "admin"),
    "ops": ("operator", "admin"),
    "admin": ("admin",),
    "audit": ("auditor", "admin"),
}


def audit_log(config: dict, record: dict) -> None:
    """审计日志：JSONL 追加写。默认只记摘要（hash），config.audit_log_content=true 才记内容。"""
    try:
        log_dir = Path(config["model_file"]).parent.parent / "state" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOCK:
            with open(log_dir / "audit.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------- Anthropic ↔ OpenAI 协议转换 ----------------

def anthropic_to_openai(body: dict) -> dict:
    msgs = list(body.get("messages", []))
    system = body.get("system")
    if system:
        msgs = [{"role": "system", "content": system}] + msgs
    return {
        "model": body.get("model", "local"),
        "messages": msgs,
        "max_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature"),
        "stream": False,  # Anthropic SSE 转换较复杂，M2 先走非流式
    }


def openai_to_anthropic(resp: dict, model: str) -> dict:
    text = ""
    if resp.get("choices"):
        text = resp["choices"][0].get("message", {}).get("content", "")
    usage = resp.get("usage", {})
    return {
        "id": resp.get("id", "msg_local"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0)},
    }


# ---------------- 网关 ----------------

class StrictHTTPServer(ThreadingHTTPServer):
    """Windows 下 SO_REUSEADDR 允许重复绑定同一端口（幽灵监听），禁用之。"""
    allow_reuse_address = False

class Gateway(BaseHTTPRequestHandler):
    config: dict = {}
    engine_ok: callable = staticmethod(lambda: False)
    engine_base: str = DEFAULT_ENGINE_BASE  # 引擎实际内部端口（动态分配）
    manager = None  # EngineManager 实例（可能为 None）

    protocol_version = "HTTP/1.1"

    # ---- 基础 ----

    def log_message(self, fmt, *a):
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _auth_role(self) -> str | None:
        """多密钥 RBAC 鉴权。返回角色名（consumer/operator/admin/auditor）或 None。

        - 本机回环地址且未配置任何密钥 → admin（个人场景默认全权）
        - config.api_keys: [{"key","name","role"}] 多虚拟密钥
        - 兼容旧字段 config.api_key（视为 admin）
        """
        host = self.config.get("host", "127.0.0.1")
        keys = self.config.get("api_keys", [])
        legacy = self.config.get("api_key")
        if host in ("127.0.0.1", "localhost", "::1"):
            if not keys and not legacy:
                return "admin"
        auth = self.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""
        for k in keys:
            if token and token == k.get("key"):
                return k.get("role", "consumer")
        if legacy and token == legacy:
            return "admin"
        return None

    def _require(self, group: str) -> str | None:
        """鉴权 + 角色校验。通过返回角色；否则已回 401/403，返回 None。"""
        role = self._auth_role()
        if role is None:
            self._json(401, {"error": {"message": "需要 API Key"}})
            return None
        if role not in ROUTE_ROLES[group]:
            self._json(403, {"error": {"message": f"角色 {role} 无权访问此操作（需要 {ROUTE_ROLES[group]}）"}})
            return None
        return role

    def _key_name(self) -> str:
        auth = self.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""
        for k in self.config.get("api_keys", []):
            if token and token == k.get("key"):
                return k.get("name", "unnamed")
        return "local"

    def _audit(self, endpoint: str, status: int, usage: dict, latency_ms: float,
               prompt_sha: str = "") -> None:
        import hashlib
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "key": self._key_name(),
            "role": self._auth_role(),
            "endpoint": endpoint,
            "model": self.config.get("name"),
            "pack_variant": self.config.get("variant"),
            "model_sha256": self.config.get("model_sha256", "")[:16],
            "status": status,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_sha256": prompt_sha[:16],
        }
        audit_log(self.config, record)
        # 按密钥核算
        name = record["key"]
        with STATS_LOCK:
            u = USAGE_BY_KEY.setdefault(name, {"requests": 0, "tokens": 0})
            u["requests"] += 1
            u["tokens"] += record["prompt_tokens"] + record["completion_tokens"]

    def _read_body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def _read_json(self) -> dict:
        """容错解析请求体（编码错误字符替换，避免 GBK/UTF-8 混杂客户端打崩）。"""
        raw = self._read_body() or b"{}"
        return json.loads(raw.decode("utf-8", "replace"))

    # ---- 引擎代理 ----

    def _proxy(self, method: str, path: str, body: bytes | None) -> None:
        import hashlib
        t0 = time.time()
        url = self.engine_base + path
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Content-Type", "application/json")
        stream = b'"stream":true' in (body or b"").replace(b" ", b"")
        prompt_sha = hashlib.sha256(body or b"").hexdigest()
        usage: dict = {}
        status = 200
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                payload = resp.read()
                status = resp.status
                with STATS_LOCK:
                    STATS["requests_total"] += 1
                    try:
                        usage = json.loads(payload).get("usage", {})
                        STATS["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        STATS["completion_tokens"] += usage.get("completion_tokens", 0)
                        t = usage.get("completion_tokens", 0)
                        tt = (resp.headers.get("X-Process-Time"))
                        if t and tt:
                            STATS["last_tps"] = round(t / float(tt), 1)
                    except Exception:
                        pass
                self.send_response(resp.status)
                ctype = "text/event-stream" if stream else "application/json; charset=utf-8"
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as e:
            payload = e.read()
            status = e.code
            with STATS_LOCK:
                STATS["errors_total"] += 1
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            status = 502
            with STATS_LOCK:
                STATS["errors_total"] += 1
            self._json(502, {"error": {"message": f"引擎不可用: {e}",
                                       "type": "engine_unreachable"}})
        finally:
            if method == "POST" and path.startswith("/v1/"):
                self._audit(path, status, usage, (time.time() - t0) * 1000, prompt_sha)

    def _relaunch_later(self, cfg_path: Path) -> None:
        """延迟 2 秒以新配置重启本网关（当前进程退出，独立进程拉起新实例）。"""
        serve_py = Path(__file__).resolve()
        if platform.system() == "Windows":
            subprocess.Popen(
                ["cmd", "/c", "timeout", "/t", "2", "/nobreak", ">nul", "&",
                 sys.executable, str(serve_py), "--config", str(cfg_path)],
                cwd=str(serve_py.parent),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0))
        else:
            subprocess.Popen(
                ["sh", "-c",
                 f"sleep 2; exec {sys.executable} {serve_py} --config '{cfg_path}'"],
                cwd=str(serve_py.parent),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        threading.Timer(1.0, lambda: os._exit(0)).start()

    # ---- 状态目录工具 ----

    def _install_dir(self) -> Path:
        return Path(self.config["model_file"]).parent.parent

    def _docs_dir(self) -> Path:
        d = self._install_dir() / "state" / "docs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- 路由 ----

    def do_GET(self):
        if self._require("view") is None:
            return
        if self.path == "/api/docs":
            docs = [{"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1)}
                    for p in sorted(self._docs_dir().iterdir())]
            return self._json(200, {"docs": docs})
        if self.path == "/api/audit":
            # 审计日志只读（auditor/admin）
            if self._require("audit") is None:
                return
            n = 100
            if "n=" in self.path:
                try:
                    n = int(self.path.split("n=")[-1])
                except ValueError:
                    pass
            log = self._install_dir() / "state" / "logs" / "audit.jsonl"
            lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
            return self._json(200, {"total": len(lines),
                                    "entries": [json.loads(x) for x in lines[-n:]]})
        if self.path == "/api/keys":
            if self._require("admin") is None:
                return
            keys = [{"name": k.get("name"), "role": k.get("role"),
                     "key_preview": k.get("key", "")[:4] + "…"}
                    for k in self.config.get("api_keys", [])]
            with STATS_LOCK:
                usage = dict(USAGE_BY_KEY)
            return self._json(200, {"keys": keys, "usage_by_key": usage})
        if self.path == "/metrics/prom":
            # Prometheus 文本 exposition 格式（Grafana 直接接）
            with STATS_LOCK:
                s = dict(STATS)
            up = 1 if self.engine_ok() else 0
            text = (
                "# HELP modelforge_requests_total Total inference requests\n"
                "# TYPE modelforge_requests_total counter\n"
                f"modelforge_requests_total {s['requests_total']}\n"
                "# HELP modelforge_prompt_tokens_total Prompt tokens processed\n"
                "# TYPE modelforge_prompt_tokens_total counter\n"
                f"modelforge_prompt_tokens_total {s['prompt_tokens']}\n"
                "# HELP modelforge_completion_tokens_total Completion tokens generated\n"
                "# TYPE modelforge_completion_tokens_total counter\n"
                f"modelforge_completion_tokens_total {s['completion_tokens']}\n"
                "# HELP modelforge_errors_total Failed requests\n"
                "# TYPE modelforge_errors_total counter\n"
                f"modelforge_errors_total {s['errors_total']}\n"
                "# HELP modelforge_uptime_seconds Gateway uptime\n"
                "# TYPE modelforge_uptime_seconds gauge\n"
                f"modelforge_uptime_seconds {round(time.time() - START_TIME, 1)}\n"
                "# HELP modelforge_engine_up Engine online status\n"
                "# TYPE modelforge_engine_up gauge\n"
                f"modelforge_engine_up {up}\n")
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", "/index.html"):
            console = PKG_DIR / "console.html"
            if console.exists():
                return self._html(console.read_bytes())
            return self._json(200, {"name": self.config.get("name"), "docs": "/v1"})
        if self.path == "/health":
            return self._json(200, {
                "status": "ok",
                "engine": "online" if self.engine_ok() else "offline",
                "uptime_s": round(time.time() - START_TIME, 1)})
        if self.path == "/branding.json":
            b = dict(self.config.get("branding", {}))
            b.setdefault("display_name", "") or b.update(
                display_name=self.config.get("name", "Local AI"))
            b["name"] = self.config.get("name")
            b["variant"] = self.config.get("variant")
            b["gpu"] = self.config.get("gpu")
            b["engine_backend"] = self.config.get("engine", {}).get("backend", "")
            b["context"] = self.config.get("engine", {}).get(
                "defaults", {}).get("context", 0)
            try:
                b["model_size_gb"] = round(
                    Path(self.config["model_file"]).stat().st_size / 1e9, 2)
            except Exception:
                b["model_size_gb"] = 0
            return self._json(200, b)
        if self.path == "/metrics":
            with STATS_LOCK:
                stats = dict(STATS)
            return self._json(200, {**stats,
                                    "engine": "online" if self.engine_ok() else "offline",
                                    "uptime_s": round(time.time() - START_TIME, 1)})
        if self.path == "/admin/diagnose":
            if self._require("ops") is None:
                return
            # 导出脱敏诊断包（硬件报告 + 日志 + 配置去密钥）
            import zipfile, io as _io, platform as _pf
            buf = _io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("config.json", json.dumps(
                    {k: v for k, v in self.config.items()
                     if k not in ("api_key", "api_keys")},
                    ensure_ascii=False, indent=2))
                zf.writestr("system.txt", f"{_pf.platform()}\npython {sys.version}")
                log = Path(self.config.get("model_file", ".")).parent.parent / \
                    "state" / "logs" / "engine.log"
                if log.exists():
                    zf.write(log, "engine.log")
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition",
                             "attachment; filename=support-bundle.zip")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/v1/models":
            if self._require("inference") is None:
                return
            # 模型名收敛为安装包名（llama.cpp 默认返回文件全路径，体验差）
            if self.engine_ok():
                try:
                    with urllib.request.urlopen(self.engine_base + "/v1/models", timeout=5) as r:
                        data = json.loads(r.read())
                    for m in data.get("data", []):
                        m["id"] = self.config.get("name", m.get("id", "local"))
                    return self._json(200, data)
                except Exception:
                    pass
            return self._json(200, {"object": "list", "data": [{
                "id": self.config.get("name", "local"), "object": "model",
                "created": int(START_TIME), "owned_by": "local"}]})
        if self.path.startswith("/v1/"):
            return self._proxy("GET", self.path, None)
        return self._json(404, {"error": {"message": f"unknown path {self.path}"}})

    def do_POST(self):
        # ---- 管理类（按角色门禁） ----
        if self.path.startswith("/admin/"):
            group = "ops" if self.path in ("/admin/restart", "/admin/update_engine") else "admin"
            if self._require(group) is None:
                return
        elif self.path.startswith("/api/") or self.path.startswith("/v1/"):
            if self._require("inference") is None:
                return
        else:
            if self._require("view") is None:
                return
        # ---- 密钥管理（admin）----
        if self.path == "/admin/keys":
            body = self._read_json()
            cfg_path = Path(self.config["_config_path"])
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            keys = cfg.setdefault("api_keys", [])
            if body.get("action") == "add":
                if body.get("role") not in ROLES:
                    return self._json(400, {"ok": False,
                                            "error": f"role 必须是 {ROLES}"})
                keys.append({"key": body["key"], "name": body.get("name", ""),
                             "role": body["role"]})
            elif body.get("action") == "del":
                keys[:] = [k for k in keys if k.get("key") != body.get("key")]
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            self.config["api_keys"] = keys
            return self._json(200, {"ok": True, "count": len(keys)})
        # ---- 文档问答（RAG）----
        if self.path.startswith("/api/docs/upload"):
            name = self.path.split("name=")[-1].split("&")[0]
            name = Path(urllib.parse.unquote(name)).name
            if not name.lower().endswith((".txt", ".md", ".docx")):
                return self._json(400, {"ok": False, "error": "仅支持 txt/md/docx"})
            (self._docs_dir() / name).write_bytes(self._read_body())
            return self._json(200, {"ok": True, "name": name})
        if self.path == "/api/docs/delete":
            body = self._read_json()
            target = self._docs_dir() / Path(body.get("name", "")).name
            target.unlink(missing_ok=True)
            return self._json(200, {"ok": True})
        if self.path == "/api/rag_search":
            sys.path.insert(0, str(PKG_DIR))
            from rag import build_index
            body = self._read_json()
            idx = build_index(self._docs_dir())
            hits = idx.search(body.get("query", ""), int(body.get("k", 4)))
            return self._json(200, {"hits": hits})
        # ---- 局域网分享开关 ----
        if self.path == "/admin/lan":
            body = self._read_json()
            enable = bool(body.get("enable"))
            key = body.get("api_key", "")
            if enable and not key:
                return self._json(400, {"ok": False,
                                        "error": "开启局域网访问必须设置 API Key"})
            cfg_path = Path(self.config["_config_path"])
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg["host"] = "0.0.0.0" if enable else "127.0.0.1"
            cfg["api_key"] = key if enable else ""
            # 保持当前实际监听端口（可能已自动顺延过），避免重启后地址漂移
            try:
                cfg["port"] = self.server.server_address[1]
            except Exception:
                pass
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            self._json(200, {"ok": True,
                             "message": "配置已更新，服务正在以新网络模式重启…"})
            self._relaunch_later(cfg_path)
            return
        # ---- 引擎更新 ----
        if self.path == "/admin/update_engine":
            mgr = self.manager
            def _update():
                try:
                    sys.path.insert(0, str(PKG_DIR))
                    from engine_manager import download_engine, find_engine_binary
                    tag = self.config.get("engine", {}).get(
                        "version_pin", {}).get("ref", "")
                    if not tag:
                        return
                    engines = self._install_dir() / "engines"
                    exe = download_engine(tag, "amd" if "amd" in
                                          self.config.get("gpu", "").lower() else "cpu",
                                          engines)
                    cfg_path = Path(self.config["_config_path"])
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    cfg["engine_binary"] = str(exe)
                    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
                    if mgr:
                        mgr.stop(); mgr.restarts = 0
                        mgr.binary = exe
                        mgr.start()
                except Exception:
                    pass
            threading.Thread(target=_update, daemon=True).start()
            return self._json(200, {"ok": True, "message": "引擎更新中，完成后自动重启"})
        # ---- 运维接口 ----
        if self.path == "/admin/uninstall":
            # 自卸载：派生独立辅助进程（杀引擎→杀网关→清自启项→删安装目录）
            install_dir = Path(self.config["model_file"]).parent.parent
            helper = Path(__file__).parent / "uninstall_helper.py"
            engine_pid = (self.manager.proc.pid
                          if self.manager and self.manager.proc else 0)
            import subprocess as _sp
            _sp.Popen(
                [__import__("sys").executable, str(helper),
                 "--dir", str(install_dir),
                 "--name", self.config.get("name", ""),
                 "--engine-pid", str(engine_pid),
                 "--gateway-pid", str(__import__("os").getpid()),
                 "--marker", str(install_dir.parent / (install_dir.name + ".uninstalled.json"))],
                cwd=str(helper.parent),
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0)
                | getattr(_sp, "DETACHED_PROCESS", 0))
            return self._json(200, {"ok": True,
                                    "message": "正在卸载：服务将停止，安装目录与自启项将被清除"})
        if self.path == "/admin/restart":
            mgr = self.manager
            if not mgr:
                return self._json(400, {"ok": False, "error": "引擎未运行（降级模式）"})
            def _restart():
                try:
                    mgr.stop()
                    mgr.restarts = 0
                    mgr.start()
                except Exception:
                    pass
            threading.Thread(target=_restart, daemon=True).start()
            return self._json(200, {"ok": True, "message": "引擎重启中，约数秒后恢复"})
        body = self._read_body()
        if self.path == "/v1/messages":
            import hashlib
            t0 = time.time()
            status = 200
            usage: dict = {}
            try:
                oai = anthropic_to_openai(json.loads(body))
            except Exception as e:
                return self._json(400, {"error": {"message": f"请求体解析失败: {e}"}})
            try:
                req = urllib.request.Request(
                    self.engine_base + "/v1/chat/completions",
                    data=json.dumps(oai).encode(), method="POST")
                req.add_header("Content-Type", "application/json")
                with urllib.request.urlopen(req, timeout=600) as resp:
                    result = json.loads(resp.read())
                usage = result.get("usage", {})
                with STATS_LOCK:
                    STATS["requests_total"] += 1
                    STATS["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    STATS["completion_tokens"] += usage.get("completion_tokens", 0)
                return self._json(200, openai_to_anthropic(result, oai["model"]))
            except Exception as e:
                status = 502
                with STATS_LOCK:
                    STATS["errors_total"] += 1
                return self._json(502, {"error": {"type": "engine_unreachable",
                                                  "message": str(e)}})
            finally:
                self._audit("/v1/messages", status, usage,
                            (time.time() - t0) * 1000,
                            hashlib.sha256(body).hexdigest())
        if self.path.startswith("/v1/"):
            return self._proxy("POST", self.path, body)
        return self._json(404, {"error": {"message": f"unknown path {self.path}"}})


# ---------------- 启动 ----------------

def main() -> int:
    ap = argparse.ArgumentParser(description="本地 AI API 网关")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["_config_path"] = str(args.config)

    # 用量核算持久化：从审计日志重建按密钥统计（重启不丢）
    try:
        audit_f = Path(config["model_file"]).parent.parent / "state" / "logs" / "audit.jsonl"
        if audit_f.exists():
            for line in audit_f.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                    u = USAGE_BY_KEY.setdefault(e.get("key", "local"),
                                                {"requests": 0, "tokens": 0})
                    u["requests"] += 1
                    u["tokens"] += e.get("prompt_tokens", 0) + e.get("completion_tokens", 0)
                except Exception:
                    continue
    except Exception:
        pass
    host = args.host or config.get("host", "127.0.0.1")
    port = args.port or int(config.get("port", 8080))
    config["host"] = host

    if host not in ("127.0.0.1", "localhost", "::1") \
            and not config.get("api_key") and not config.get("api_keys"):
        print("[拒绝启动] 绑定非本机地址必须在 config.json 中设置 api_key 或 api_keys")
        return 1

    # 引擎看门狗
    manager = None
    engine_bin = config.get("engine_binary")
    if engine_bin and Path(engine_bin).exists():
        sys.path.insert(0, str(PKG_DIR))
        from engine_manager import EngineManager, build_engine_args, find_free_port
        eng_port = find_free_port()
        eng_args = build_engine_args(config, float(config.get("gpu_free_vram_gb", 0)),
                                     port=eng_port)
        manager = EngineManager(
            Path(engine_bin), eng_args,
            args.config.parent.parent / "state" / "logs" / "engine.log",
            port=eng_port)
        try:
            manager.start()
            Gateway.engine_base = f"http://127.0.0.1:{eng_port}"
            print(f"  引擎已启动（内部端口 {eng_port}）")
        except Exception as e:
            print(f"  [警告] 引擎启动失败: {e}（网关以降级模式运行）")
            manager = None
        if manager:
            def watchdog():
                while True:
                    time.sleep(10)
                    manager.ensure_running()
            threading.Thread(target=watchdog, daemon=True).start()
    else:
        print("  [降级模式] 未配置引擎二进制，仅 /health /metrics /v1/models 可用")

    Gateway.config = config
    Gateway.engine_ok = staticmethod(
        lambda: manager.is_healthy() if manager else False)
    Gateway.manager = manager

    # 端口自动顺延：Windows 保留端口段等原因导致绑定失败时尝试后续端口
    server = None
    actual_port = port
    for candidate in range(port, port + 21):
        try:
            server = StrictHTTPServer((host, candidate), Gateway)
            actual_port = candidate
            break
        except (OSError, PermissionError):
            continue
    if server is None:
        print(f"[失败] 端口 {port}~{port + 20} 均不可用")
        return 1
    if actual_port != port:
        print(f"  [提示] 端口 {port} 被占用/保留，已自动顺延到 {actual_port}")
    # 记录实际端口，供安装向导/打包台发现
    try:
        (args.config.parent / "runtime.json").write_text(json.dumps(
            {"port": actual_port, "pid": __import__("os").getpid()}))
    except Exception:
        pass

    print(f"[{config.get('name')}] API 网关已启动: http://{host}:{actual_port}")
    print(f"  控制台:   http://{host}:{actual_port}/")
    print(f"  OpenAI:   http://{host}:{actual_port}/v1")
    print(f"  Anthropic: http://{host}:{actual_port}/v1/messages")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if manager:
            manager.stop()
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
