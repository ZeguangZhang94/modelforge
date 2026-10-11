#!/usr/bin/env python3
"""ModelForge 模型宿主（hub）：一个页面管理本机所有已安装模型。

- 自动扫描安装根目录（~/.modelforge、%LOCALAPPDATA%、~/.local/share）下的安装实例
- 每个模型卡片：状态（运行中/已停止）、档位、显卡、端口
- 操作：启动 / 停止 / 打开控制台 / 定位目录

用法：python -m modelforge hub [--port 18092]
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def install_roots() -> list[Path]:
    roots = [Path.home() / ".modelforge"]
    if platform.system() == "Windows":
        roots.append(Path(os.environ.get("LOCALAPPDATA", Path.home())))
    else:
        roots.append(Path.home() / ".local" / "share")
    return roots


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if platform.system() == "Windows":
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                               capture_output=True, text=True, timeout=10,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return str(pid) in r.stdout
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def scan_installs() -> list[dict]:
    """扫描所有安装实例。"""
    installs = []
    seen = set()
    for root in install_roots():
        if not root.exists():
            continue
        for cfg in root.glob("*/state/config.json"):
            try:
                c = json.loads(cfg.read_text(encoding="utf-8"))
            except Exception:
                continue
            name = c.get("name", cfg.parent.parent.name)
            if name in seen:
                continue
            seen.add(name)
            runtime_f = cfg.parent / "runtime.json"
            port, pid = c.get("port", 8080), 0
            if runtime_f.exists():
                try:
                    rt = json.loads(runtime_f.read_text())
                    port, pid = rt.get("port", port), rt.get("pid", 0)
                except Exception:
                    pass
            running = pid_alive(pid)
            if running:  # 二次确认端口真活着
                try:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=1.5)
                except Exception:
                    running = False
            installs.append({
                "name": name,
                "display_name": c.get("branding", {}).get("display_name") or name,
                "logo": c.get("branding", {}).get("logo_text", "⚡"),
                "variant": c.get("variant", ""),
                "gpu": c.get("gpu", ""),
                "port": port, "pid": pid, "running": running,
                "install_dir": str(cfg.parent.parent),
                "pkg_dir": c.get("pkg_dir", ""),
                "config": str(cfg),
            })
    return installs


PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ModelForge 模型宿主</title>
<style>
  body { background:#0d1117; color:#e6edf3; font:14px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
         max-width:860px; margin:36px auto; padding:0 20px; }
  h1 { font-size:20px; } .dim { color:#8b949e; font-size:12.5px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:10px;
          padding:14px 18px; margin-top:12px; display:flex; align-items:center; gap:14px; }
  .logo { font-size:28px; }
  .info { flex:1; } .info b { font-size:15px; }
  .badge { font-size:11.5px; padding:1px 8px; border-radius:10px; }
  .on { background:#0d2818; color:#3fb950; border:1px solid #3fb950; }
  .off { background:#21262d; color:#8b949e; border:1px solid #30363d; }
  button { background:#4c9aff; color:#fff; border:0; border-radius:6px;
           padding:6px 14px; font:inherit; font-size:13px; cursor:pointer; margin-left:6px; }
  button.ghost { background:#21262d; border:1px solid #30363d; color:#e6edf3; }
  button:disabled { opacity:.4; }
</style></head><body>
<h1>🗂 ModelForge 模型宿主</h1>
<div class="dim">管理本机所有已安装的本地 AI 模型。<span id="count"></span></div>
<div id="list"></div>
<div id="empty" class="dim" style="display:none;margin-top:20px">
  还没有已安装的模型。先打开打包台（python -m modelforge gui）制作并安装一个。</div>
<script>
const $ = id => document.getElementById(id);
async function load() {
  const d = await (await fetch('/api/list')).json();
  $('count').textContent = `（共 ${d.installs.length} 个）`;
  $('empty').style.display = d.installs.length ? 'none' : 'block';
  $('list').innerHTML = d.installs.map(m => `
    <div class="card">
      <div class="logo">${m.logo}</div>
      <div class="info"><b>${m.display_name}</b>
        <span class="badge ${m.running ? 'on' : 'off'}">${m.running ? '运行中 :' + m.port : '已停止'}</span>
        <div class="dim">${m.variant} · ${m.gpu}</div>
      </div>
      ${m.running
        ? `<button onclick="window.open('http://127.0.0.1:${m.port}/','_blank')">打开控制台</button>
           <button class="ghost" onclick="act('stop','${m.name}')">停止</button>`
        : `<button onclick="act('start','${m.name}')" ${m.pkg_dir ? '' : 'disabled title="缺少 pkg_dir，请从安装包 START-HERE 启动"'}>启动</button>`}
    </div>`).join('');
}
async function act(action, name) {
  await fetch('/api/' + action, {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name})});
  setTimeout(load, action === 'start' ? 4000 : 1500);
}
load(); setInterval(load, 5000);
</script></body></html>"""


class StrictHTTPServer(ThreadingHTTPServer):
    """Windows 下 SO_REUSEADDR 允许重复绑定同一端口（幽灵监听），禁用之。"""
    allow_reuse_address = False

class HubHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/list":
            return self._json(200, {"installs": scan_installs()})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except Exception:
            return self._json(400, {"ok": False})
        target = next((m for m in scan_installs() if m["name"] == body.get("name")), None)
        if not target:
            return self._json(404, {"ok": False, "error": "未找到该模型"})

        if self.path == "/api/start":
            pkg = Path(target["pkg_dir"]) if target["pkg_dir"] else None
            if not pkg or not (pkg / "serve.py").exists():
                return self._json(400, {"ok": False,
                                        "error": "找不到安装包目录，无法启动"})
            log = open(Path(target["install_dir"]) / "state" / "serve.log", "ab")
            subprocess.Popen(
                [sys.executable, str(pkg / "serve.py"), "--config", target["config"]],
                cwd=str(pkg), stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0))
            return self._json(200, {"ok": True})

        if self.path == "/api/stop":
            if target["pid"]:
                try:
                    if platform.system() == "Windows":
                        subprocess.run(["taskkill", "/PID", str(target["pid"]), "/T", "/F"],
                                       capture_output=True, timeout=15,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    else:
                        os.kill(target["pid"], 15)
                except Exception as e:
                    return self._json(500, {"ok": False, "error": str(e)})
            return self._json(200, {"ok": True})

        return self._json(404, {"ok": False})


def run_hub(port: int = 18092) -> int:
    server = StrictHTTPServer(("127.0.0.1", port), HubHandler)
    print(f"模型宿主已启动: http://127.0.0.1:{port}  （Ctrl+C 停止）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
