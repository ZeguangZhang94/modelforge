#!/usr/bin/env python3
"""ModelForge 打包台：全流程 Web 界面，零命令行。

三个工作区（单页）：
1. 获取模型 —— 输入 ModelScope/HuggingFace 仓库 ID，列出 GGUF 文件，
   勾选档位后台下载（断点续传 + 进度条）
2. 分析打包 —— 分析模型 → 硬件需求预估 → 生成安装包 zip
3. 安装启动 —— 本机一键安装刚生成的包并启动服务，直达控制台

用法：python -m modelforge gui [--port 18090]
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import uuid
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .analyzer import analyze
from .builder import assemble, build_pack, zip_package
from .estimator import estimate_all, print_estimate_table

ROOT = Path.cwd()
MODELS_DIR = ROOT / "models_download"
DIST_DIR = ROOT / "dist"

# 下载任务表：job_id → {file, total, done, status, error}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

REPO_API = {
    "modelscope": "https://modelscope.cn/api/v1/models/{repo}/repo/files",
    "huggingface": "https://hf-mirror.com/api/models/{repo}",  # 国内镜像
}
DOWNLOAD_URL = {
    "modelscope": "https://modelscope.cn/models/{repo}/resolve/master/{file}",
    "huggingface": "https://hf-mirror.com/{repo}/resolve/main/{file}",
}


# ---------------- 模型仓库查询与下载 ----------------

def list_repo_files(source: str, repo: str) -> list[dict]:
    """列出仓库中的 GGUF 文件（含子目录，如 BF16/MTP 目录内的文件）。"""
    files: list[dict] = []
    if source == "modelscope":
        def walk(path: str = "") -> None:
            url = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true"
            req = urllib.request.Request(
                url, headers={"User-Agent": "modelforge"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
            for f in data.get("Data", {}).get("Files", []):
                if f.get("Path", "").lower().endswith(".gguf"):
                    files.append({"path": f["Path"],
                                  "size_gb": round(f.get("Size", 0) / 1e9, 2)})
        walk()
    else:
        req = urllib.request.Request(
            REPO_API["huggingface"].format(repo=repo),
            headers={"User-Agent": "modelforge"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        for s in data.get("siblings", []):
            name = s.get("rfilename", "")
            if name.lower().endswith(".gguf"):
                files.append({"path": name, "size_gb": round((s.get("size") or 0) / 1e9, 2)})
    return sorted(files, key=lambda x: x["path"])


def download_job(source: str, repo: str, files: list[str], dest: Path, job_id: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
    try:
        for fname in files:
            url = DOWNLOAD_URL[source].format(repo=repo, file=fname)
            out = dest / Path(fname).name
            part = out.with_suffix(out.suffix + ".part")
            headers, mode = {}, "wb"
            if part.exists():
                headers["Range"] = f"bytes={part.stat().st_size}-"
                mode = "ab"
            req = urllib.request.Request(
                url, headers={**headers, "User-Agent": "modelforge"})
            resp = urllib.request.urlopen(req, timeout=60)
            if mode == "ab" and resp.status != 206:  # 服务端不接受续传则重来
                resp.close()
                resp = urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": "modelforge"}),
                    timeout=60)
                mode = "wb"
            total = int(resp.headers.get("Content-Length", 0)) + \
                (part.stat().st_size if mode == "ab" else 0)
            with JOBS_LOCK:
                JOBS[job_id].update(file=fname, total=total,
                                    done=part.stat().st_size if mode == "ab" else 0)
            with resp, open(part, mode) as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    with JOBS_LOCK:
                        JOBS[job_id]["done"] += len(chunk)
            part.rename(out)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)


# ---------------- 页面 ----------------

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ModelForge 打包台</title>
<style>
  body { background:#0d1117; color:#e6edf3; font:14px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
         max-width:820px; margin:32px auto; padding:0 20px; }
  h1 { font-size:20px; } h2 { font-size:15px; margin:26px 0 4px; color:#4c9aff; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px 18px; margin-top:10px; }
  label { color:#8b949e; font-size:12.5px; display:block; margin-top:10px; }
  input,select { width:100%; background:#0d1117; border:1px solid #30363d; color:#e6edf3;
          border-radius:8px; padding:8px 12px; font:inherit; margin-top:4px; box-sizing:border-box; }
  .row { display:flex; gap:12px; } .row > div { flex:1; }
  button { background:#4c9aff; color:#fff; border:0; border-radius:8px;
           padding:9px 20px; font:inherit; cursor:pointer; margin-top:12px; }
  button.ghost { background:#21262d; border:1px solid #30363d; }
  button:disabled { opacity:.4; }
  pre { background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:12px;
        margin-top:12px; white-space:pre-wrap; word-break:break-all; font-size:12.5px;
        max-height:340px; overflow:auto; }
  .ok { border-color:#3fb950 !important; } .err { border-color:#f85149 !important; }
  .file { display:flex; align-items:center; gap:8px; padding:4px 0; }
  .file input { width:auto; margin:0; }
  .bar { height:8px; background:#21262d; border-radius:4px; margin-top:8px; overflow:hidden; }
  .bar > div { height:100%; background:#3fb950; width:0; transition:width .3s; }
  .dim { color:#8b949e; font-size:12.5px; }
  a { color:#4c9aff; }
</style></head><body>
<h1>⚡ ModelForge 打包台</h1>
<div class="dim">从下载模型到生成安装包、本机安装启动，全部在此页面完成。</div>

<h2>① 获取模型</h2>
<div class="card">
  <div class="dim" style="margin-bottom:8px">模型商店（点击卡片自动填入，推荐档位已预选）：</div>
  <div id="catalog" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px"></div>
  <details style="margin-top:12px"><summary class="dim" style="cursor:pointer">自定义仓库（高级）</summary>
  <div class="row">
    <div><label>来源</label>
      <select id="src"><option value="modelscope">ModelScope（国内快）</option>
      <option value="huggingface">HuggingFace（镜像）</option></select></div>
    <div><label>仓库 ID（如 unsloth/Qwen3.8-27B-GGUF）</label>
      <input id="repo" placeholder="组织/仓库名"></div>
  </div>
  <button class="ghost" onclick="listFiles()">列出 GGUF 文件</button>
  </details>
  <div id="files" style="margin-top:10px"></div>
  <div id="dlarea" style="display:none">
    <button onclick="startDownload()">下载选中文件</button>
    <div class="bar"><div id="dlbar"></div></div>
    <div class="dim" id="dltext"></div>
  </div>
</div>

<h2>② 分析并打包</h2>
<div class="card">
  <label>模型目录（下载完成后自动填入，也可手动填或从本地已有目录选择）</label>
  <input id="dir" list="localdirs" placeholder="D:\models\qwen38-27b">
  <datalist id="localdirs"></datalist>
  <div class="row">
    <div><label>安装包名称 *</label><input id="name" placeholder="qwen38-27b-local"></div>
    <div><label>版本号</label><input id="ver" value="1.0.0"></div>
  </div>
  <div class="row">
    <div><label>发布者</label><input id="pub"></div>
    <div><label>引擎版本锁定</label><input id="tag" value="b11526"></div>
    <div><label>上下文</label><input id="ctx" value="8192"></div>
  </div>
  <div class="row">
    <div><label>界面显示名（可空）</label><input id="dname" placeholder="Qwen3.8 27B 本地版"></div>
    <div><label>Logo（emoji）</label><input id="logo" value="⚡"></div>
    <div><label>主题色</label><input id="color" type="color" value="#4c9aff"
      style="padding:2px;height:38px"></div>
  </div>
  <label>控制台欢迎语（可空）</label><input id="welcome" placeholder="欢迎使用本地 AI，开始对话吧！">
  <label>自定义许可证文案（可空，默认读模型自带 LICENSE）</label><input id="lictext" placeholder="例如：仅供内部使用，禁止再分发">
  <label style="margin-top:10px"><input type="checkbox" id="bundle" checked style="width:auto">
    内置推理引擎（打进安装包，终端离线可装；引擎缓存在 tests/engine_cache）</label>
  <br>
  <button class="ghost" onclick="run('analyze')">分析模型</button>
  <button onclick="run('build')">生成安装包</button>
  <pre id="out" style="display:none"></pre>
</div>

<h2>③ 本机安装并启动</h2>
<div class="card">
  <label>安装包目录（打包完成后自动填入）</label>
  <input id="pkg" placeholder="D:\modelforge\dist\xxx-1.0.0">
  <button onclick="installStart()" id="installbtn">一键安装并启动</button>
  <pre id="installout" style="display:none"></pre>
  <div id="donelink"></div>
</div>

<script>
const $ = id => document.getElementById(id);
let jobId = null, pkgDir = '';

async function api(path, body) {
  const r = await fetch('/api/' + path, {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  return r.json();
}

async function listFiles() {
  $('files').innerHTML = '<span class="dim">查询仓库文件中…</span>';
  const d = await api('repo_files', {source:$('src').value, repo:$('repo').value.trim()});
  if (!d.ok) { $('files').innerHTML = '<span class="dim">查询失败: '+d.error+'</span>'; return; }
  if (!d.files.length) { $('files').innerHTML = '<span class="dim">该仓库没有 .gguf 文件</span>'; return; }
  $('files').innerHTML = d.files.map((f,i) =>
    `<div class="file"><input type="checkbox" id="f${i}" value="${f.path}">
     <label for="f${i}" style="margin:0">${f.path}（${f.size_gb} GB）</label></div>`).join('');
  $('dlarea').style.display = 'block';
}

async function startDownload() {
  const files = [...document.querySelectorAll('#files input:checked')].map(c => c.value);
  if (!files.length) { alert('先勾选要下载的档位文件'); return; }
  const d = await api('download', {source:$('src').value, repo:$('repo').value.trim(), files});
  jobId = d.job_id;
  $('dir').value = d.dest;
  $('name').value = $('repo').value.split('/').pop().replace(/-GGUF$/i,'').toLowerCase() + '-local';
  poll();
}

async function poll() {
  if (!jobId) return;
  const d = await (await fetch('/api/progress?job=' + jobId)).json();
  const pct = d.total ? Math.round(d.done / d.total * 100) : 0;
  $('dlbar').style.width = pct + '%';
  $('dltext').textContent = d.status === 'done' ? '✔ 下载完成，可去 ② 分析打包'
    : d.status === 'error' ? '✘ 下载失败: ' + d.error
    : `${d.file}  ${(d.done/1e9).toFixed(2)} / ${(d.total/1e9).toFixed(2)} GB（${pct}%）`;
  if (d.status === 'running') setTimeout(poll, 1000);
}

async function run(action) {
  const out = $('out'); out.style.display = 'block'; out.className = '';
  out.textContent = action === 'analyze' ? '分析中…' : '打包中（大文件复制需要时间，请耐心）…';
  const d = await api(action, {
    model_dir: $('dir').value.trim(), name: $('name').value.trim(),
    version: $('ver').value.trim(), publisher: $('pub').value.trim(),
    engine_tag: $('tag').value.trim(), context: parseInt($('ctx').value) || 8192,
    display_name: $('dname').value.trim(), theme_color: $('color').value,
    logo_text: $('logo').value.trim() || '⚡',
    welcome: $('welcome').value.trim(), license_text: $('lictext').value.trim(),
    bundle_engines: $('bundle').checked ? 'tests/engine_cache' : ''});
  out.textContent = d.output;
  out.className = d.ok ? 'ok' : 'err';
  if (d.ok && action === 'build' && d.pkg_dir) { $('pkg').value = d.pkg_dir; pkgDir = d.pkg_dir; }
}

// 页面加载时列出本地已下载的模型目录
fetch('/api/local_dirs').then(r => r.json()).then(d => {
  $('localdirs').innerHTML = (d.dirs || []).map(p => `<option value="${p}">`).join('');
});

// 模型商店卡片
let CATALOG = [];
fetch('/api/catalog').then(r => r.json()).then(d => {
  CATALOG = d.models || [];
  $('catalog').innerHTML = CATALOG.map((m, i) => `
    <div onclick="pickModel(${i})" style="background:#0d1117;border:1px solid #30363d;border-radius:8px;
      padding:10px;cursor:pointer" onmouseover="this.style.borderColor='#4c9aff'"
      onmouseout="this.style.borderColor='#30363d'">
      <div style="font-size:20px">${m.logo} <b style="font-size:13px">${m.name}</b></div>
      <div class="dim" style="font-size:11.5px;margin-top:4px">${m.params} · 显存≥${m.vram_gb}GB</div>
      <div class="dim" style="font-size:11.5px">${m.desc}</div>
      <div style="margin-top:4px">${m.tags.map(t =>
        `<span style="font-size:10.5px;border:1px solid #30363d;border-radius:4px;padding:0 4px;margin-right:3px">${t}</span>`).join('')}</div>
    </div>`).join('');
});

async function pickModel(i) {
  const m = CATALOG[i];
  const useMs = !!m.repo_ms;
  $('src').value = useMs ? 'modelscope' : 'huggingface';
  $('repo').value = useMs ? m.repo_ms : m.repo_hf;
  $('name').value = m.id + '-local';
  $('dname').value = m.name + ' 本地版';
  $('logo').value = m.logo;
  await listFiles();
  // 自动勾选推荐档位
  setTimeout(() => {
    const boxes = [...document.querySelectorAll('#files input')];
    const hit = boxes.find(b => b.value.toLowerCase().includes(m.quant));
    if (hit) hit.checked = true;
  }, 300);
}

async function installStart() {
  const out = $('installout'); out.style.display = 'block'; out.className = '';
  out.textContent = '安装中（复制模型 + 自检引擎，大模型需要几分钟）…';
  $('installbtn').disabled = true;
  const d = await api('install_start', {pkg_dir: $('pkg').value.trim()});
  out.textContent = d.output;
  out.className = d.ok ? 'ok' : 'err';
  if (d.ok && d.console_url) {
    $('donelink').innerHTML = `<p>✔ 服务已启动 → <a href="${d.console_url}" target="_blank">打开控制台</a>
      （API: ${d.console_url}v1）</p>`;
  }
  $('installbtn').disabled = false;
}
</script></body></html>"""


# ---------------- HTTP 服务 ----------------

class StrictHTTPServer(ThreadingHTTPServer):
    """Windows 下 SO_REUSEADDR 允许重复绑定同一端口（幽灵监听），禁用之。"""
    allow_reuse_address = False

class GuiHandler(BaseHTTPRequestHandler):
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
        if self.path.startswith("/api/progress"):
            job = self.path.split("job=")[-1]
            with JOBS_LOCK:
                d = dict(JOBS.get(job, {"status": "unknown"}))
            return self._json(200, d)
        if self.path == "/api/local_dirs":
            dirs = [str(p) for p in MODELS_DIR.iterdir() if p.is_dir()] \
                if MODELS_DIR.exists() else []
            for extra in (Path("D:/models"), Path("D:/ai-models")):
                if extra.exists():
                    dirs += [str(p) for p in extra.iterdir() if p.is_dir()]
            return self._json(200, {"dirs": dirs})
        if self.path == "/api/catalog":
            from .catalog import CATALOG
            return self._json(200, {"models": CATALOG})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except Exception:
            return self._json(400, {"ok": False, "error": "bad json"})
        handlers = {
            "/api/repo_files": self._repo_files,
            "/api/download": self._download,
            "/api/analyze": self._analyze,
            "/api/build": self._build,
            "/api/install_start": self._install_start,
        }
        fn = handlers.get(self.path)
        if not fn:
            return self._json(404, {"ok": False, "error": "unknown api"})
        try:
            return self._json(200, fn(body))
        except Exception:
            return self._json(200, {"ok": False,
                                    "output": "失败:\n" + traceback.format_exc(limit=3)})

    # ---- 各接口 ----

    @staticmethod
    def _repo_files(body: dict) -> dict:
        try:
            files = list_repo_files(body["source"], body["repo"])
            return {"ok": True, "files": files}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @staticmethod
    def _download(body: dict) -> dict:
        dest = MODELS_DIR / body["repo"].replace("/", "_")
        job_id = uuid.uuid4().hex[:8]
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "queued", "file": "", "total": 0, "done": 0}
        threading.Thread(target=download_job,
                         args=(body["source"], body["repo"], body["files"], dest, job_id),
                         daemon=True).start()
        return {"ok": True, "job_id": job_id, "dest": str(dest)}

    @staticmethod
    def _analyze(body: dict) -> dict:
        profile = analyze(Path(body["model_dir"]))
        table = print_estimate_table(profile, estimate_all(profile, body.get("context", 8192)))
        return {"ok": True, "output": table +
                f"\n\n能力: {', '.join(profile.capabilities)}  引擎后端: {profile.engine_backend}"
                f"  许可证: {profile.license}\n\n确认无误后点「生成安装包」"}

    @staticmethod
    def _build(body: dict) -> dict:
        if not body.get("name"):
            return {"ok": False, "output": "请填写安装包名称"}
        from .builder import bundle_engines
        from .modelpack import Branding
        branding = Branding(
            display_name=body.get("display_name", ""),
            logo_text=body.get("logo_text") or "⚡",
            theme_color=body.get("theme_color") or "#4c9aff",
            welcome_message=body.get("welcome", ""),
            license_text=body.get("license_text", ""),
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            pack, profile = build_pack(
                model_dir=body["model_dir"], name=body["name"],
                version=body.get("version", "1.0.0"),
                publisher=body.get("publisher", ""),
                context=body.get("context", 8192),
                engine_tag=body.get("engine_tag", ""),
                branding=branding)
            print(print_estimate_table(profile, estimate_all(profile)))
            root = assemble(pack, body["model_dir"], DIST_DIR)
            if body.get("bundle_engines"):
                n = bundle_engines(root, body["bundle_engines"])
                print(f"\n已内置引擎文件 {n} 个")
            zip_path = zip_package(root)
            print(f"\n✔ 打包完成\n包目录: {root}\n分发文件: {zip_path}")
        return {"ok": True, "output": buf.getvalue(), "pkg_dir": str(root)}

    @staticmethod
    def _install_start(body: dict) -> dict:
        """本机一键安装（幂等）+ 后台启动网关。"""
        pkg = Path(body["pkg_dir"])
        installer = pkg / "installer.py"
        if not installer.exists():
            return {"ok": False, "output": f"安装包目录无效: {pkg}"}
        pack = json.loads((pkg / "modelpack.json").read_text(encoding="utf-8"))
        name = pack["name"]
        install_dir = Path.home() / ".modelforge" / name
        buf = io.StringIO()
        with redirect_stdout(buf):
            import os as _os
            env = {**_os.environ, "PYTHONIOENCODING": "utf-8"}  # 子进程输出统一 UTF-8
            r = subprocess.run(
                [sys.executable, str(installer), "--yes", "--dir", str(install_dir)],
                capture_output=True, timeout=1800, cwd=str(pkg), env=env)
            print(r.stdout.decode("utf-8", "replace")[-3000:])
            if r.returncode != 0:
                print(r.stderr.decode("utf-8", "replace")[-1500:])
                return {"ok": False, "output": buf.getvalue()}
            # 后台启动网关（独立进程，不随打包台退出）
            config = install_dir / "state" / "config.json"
            port = pack["service"]["port"]
            subprocess.Popen(
                [sys.executable, str(pkg / "serve.py"), "--config", str(config)],
                cwd=str(pkg), stdout=open(install_dir / "state" / "serve.log", "ab"),
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0))
            time.sleep(1)
            # 读网关实际端口（可能自动顺延）
            actual_port = port
            runtime_f = install_dir / "state" / "runtime.json"
            for _ in range(20):
                time.sleep(0.5)
                if runtime_f.exists():
                    try:
                        actual_port = json.loads(runtime_f.read_text())["port"]
                        break
                    except Exception:
                        pass
            print(f"\n网关启动中，控制台: http://127.0.0.1:{actual_port}/")
        return {"ok": True, "output": buf.getvalue(),
                "console_url": f"http://127.0.0.1:{actual_port}/"}


def run_gui(port: int = 18090) -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    DIST_DIR.mkdir(exist_ok=True)
    server = StrictHTTPServer(("127.0.0.1", port), GuiHandler)
    print(f"打包台已启动: http://127.0.0.1:{port}  （Ctrl+C 停止）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
