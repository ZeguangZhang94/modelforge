#!/usr/bin/env python3
"""生成 ModelForge 使用手册 PDF（ReportLab 路线，中文 CJK 字体）。"""

import os
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, Preformatted)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

regular_font = os.environ.get("DAIMON_CJK_FONT_REGULAR")
bold_font = os.environ.get("DAIMON_CJK_FONT_BOLD")
assert regular_font and bold_font, "缺少受管 CJK 字体"
pdfmetrics.registerFont(TTFont("CJK", regular_font))
pdfmetrics.registerFont(TTFont("CJK-Bold", bold_font))
pdfmetrics.registerFontFamily("CJK", normal="CJK", bold="CJK-Bold",
                              italic="CJK", boldItalic="CJK-Bold")

OUT = Path(__file__).parent.parent / "dist" / "ModelForge使用手册.pdf"

styles = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=styles["Title"], fontName="CJK-Bold",
                            fontSize=26, leading=34, textColor=HexColor("#1a1a2e")),
    "subtitle": ParagraphStyle("st", parent=styles["Normal"], fontName="CJK",
                               fontSize=13, leading=20, textColor=HexColor("#555555")),
    "h1": ParagraphStyle("h1", parent=styles["Heading1"], fontName="CJK-Bold",
                         fontSize=17, leading=24, spaceBefore=18, spaceAfter=8,
                         textColor=HexColor("#1a1a2e")),
    "h2": ParagraphStyle("h2", parent=styles["Heading2"], fontName="CJK-Bold",
                         fontSize=13.5, leading=19, spaceBefore=12, spaceAfter=6,
                         textColor=HexColor("#2d2d44")),
    "body": ParagraphStyle("b", parent=styles["Normal"], fontName="CJK",
                           fontSize=10.5, leading=17, textColor=HexColor("#333333")),
    "code": ParagraphStyle("c", fontName="CJK", fontSize=9, leading=13,
                           backColor=HexColor("#f5f5f5"), textColor=HexColor("#222222")),
    "caption": ParagraphStyle("cap", fontName="CJK", fontSize=9,
                              textColor=HexColor("#666666")),
}


def table(data, widths):
    cell = ParagraphStyle("cell", fontName="CJK", fontSize=9.5, leading=13,
                          textColor=HexColor("#333333"))
    head = ParagraphStyle("cellh", fontName="CJK-Bold", fontSize=9.5, leading=13,
                          textColor=HexColor("#1a1a2e"))
    wrapped = [[Paragraph(str(c), head) if i == 0 else Paragraph(str(c), cell)
                for c in row] for i, row in enumerate(data)]
    t = Table(wrapped, colWidths=widths)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "CJK"),
        ("FONTNAME", (0, 0), (-1, 0), "CJK-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("LINEABOVE", (0, 0), (-1, 0), 1.5, HexColor("#000000")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, HexColor("#000000")),
        ("LINEBELOW", (0, -1), (-1, -1), 1.5, HexColor("#000000")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFont("CJK", 8.5)
    canvas.setFillColor(HexColor("#888888"))
    canvas.drawCentredString(w / 2, h - 1.4 * cm, "ModelForge · 模型熔炉 — 使用手册")
    canvas.drawCentredString(w / 2, 1.2 * cm, f"第 {doc.page} 页")
    canvas.restoreState()


def cover(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(HexColor("#4c9aff"))
    canvas.setLineWidth(2)
    canvas.line(4 * cm, h - 6.5 * cm, w - 4 * cm, h - 6.5 * cm)
    canvas.line(4 * cm, 5.5 * cm, w - 4 * cm, 5.5 * cm)
    canvas.restoreState()


story = []
# ---- 封面 ----
story.append(Spacer(1, 5 * cm))
story.append(Paragraph("ModelForge · 模型熔炉", S["title"]))
story.append(Spacer(1, 0.6 * cm))
story.append(Paragraph("把任何一个开源 AI 模型，变成小白也能双击安装的本地 AI 服务安装包",
                       S["subtitle"]))
story.append(Spacer(1, 1.2 * cm))
story.append(Paragraph("使用手册 · v1.0 · 2026 年 10 月", S["subtitle"]))
story.append(PageBreak())

# ---- 1 产品简介 ----
story.append(Paragraph("1. 产品简介", S["h1"]))
story.append(Paragraph(
    "ModelForge 是一个两层结构的本地 AI 分发系统：打包者在「打包台」里选择模型、"
    "填写安装包名称，即可生成该模型专属的安装包；终端用户拿到安装包后双击即用——"
    "自动识别显卡、自动匹配量化档位、自动充分利用显存，装完直接得到 "
    "OpenAI / Anthropic 兼容的本地 API 和一个全中文网页控制台。", S["body"]))
story.append(Spacer(1, 6))
story.append(Paragraph("真机实测（AMD RX 7900 XTX 24GB + 68GB RAM，Windows 11）：", S["body"]))
story.append(table([
    ["模型", "档位", "实测输出速度"],
    ["Qwen2.5 0.5B", "Q4_K_M", "259 tokens/s"],
    ["Qwen2.5 1.5B", "Q4_K_M", "327 tokens/s"],
    ["Qwen3.8 27B（MTP 投机解码）", "UD-Q4_K_M", "43.7 tokens/s"],
], [6 * cm, 4 * cm, 4.5 * cm]))
story.append(Paragraph("表 1：真实硬件推理速度实测", S["caption"]))
story.append(PageBreak())

# ---- 2 打包者 ----
story.append(Paragraph("2. 打包者：从模型到安装包", S["h1"]))
story.append(Paragraph("启动打包台后，全部操作在浏览器页面中完成：", S["body"]))
story.append(Preformatted("python -m modelforge gui        # 浏览器打开 http://127.0.0.1:18090",
                          S["code"]))
story.append(Paragraph("2.1 获取模型（模型商店）", S["h2"]))
story.append(Paragraph(
    "打包台内置模型商店：17 个经过存在性验证的精选模型（Qwen2.5 / Qwen3 / Qwen3.8 / "
    "Llama 3.1 / DeepSeek R1 / Coder / VL 等），卡片标注参数量、显存需求与推荐档位。"
    "点击卡片自动填入仓库信息并预选推荐量化档，点「下载选中文件」即可（断点续传）。"
    "模型已在本地时可直接选择本地目录。", S["body"]))
story.append(Paragraph("2.2 分析与打包", S["h2"]))
story.append(Paragraph(
    "填安装包名称后点「分析模型」查看硬件需求预估表，确认后点「生成安装包」。"
    "可定制品牌：显示名、Logo、主题色、欢迎语、自定义许可证文案；"
    "勾选「内置推理引擎」与「内置 Python 运行时」后，安装包完全离线可用、"
    "终端用户连 Python 都不需要装。", S["body"]))
story.append(Paragraph("2.3 Linux 分发（Docker Model Runner）", S["h2"]))
story.append(Preformatted(
    "python -m modelforge build-dmr --name qwen38-27b --repo unsloth/Qwen3.8-27B-GGUF",
    S["code"]))
story.append(Paragraph(
    "Linux 平台走 Docker 生态：生成的安装包通过 docker model pull 拉取模型（OCI 制品），"
    "自动检测 docker 与插件、给出 OpenAI 兼容 API 地址（localhost:12434/v1）。", S["body"]))
story.append(PageBreak())

# ---- 3 终端用户 ----
story.append(Paragraph("3. 终端用户：安装与日常使用", S["h1"]))
story.append(Paragraph("3.1 安装（四步）", S["h2"]))
story.append(Paragraph(
    "Windows 解压 zip 后双击 START-HERE.bat（Linux 运行 ./install.sh），浏览器自动打开安装向导："
    "① 看硬件检测（显卡已自动选好）→ ② 看许可证 → ③ 选档位（推荐项已勾选，"
    "硬件不足的档位自动禁选）→ ④ 开始安装 → 启动服务并打开控制台。"
    "勾选「登录后自动启动」则以后开机即用（免管理员）。端口被占用时自动顺延。", S["body"]))
story.append(Paragraph("3.2 控制台功能", S["h2"]))
story.append(table([
    ["区域", "功能"],
    ["左侧-状态", "引擎在线状态、运行时长、会话 token 吞吐（2 秒刷新）"],
    ["左侧-模型信息", "模型名、量化档位、显卡、大小、上下文、引擎后端"],
    ["左侧-文档问答", "上传 txt/md/docx，勾选后回答自动参考文档内容"],
    ["左侧-历史对话", "对话自动保存，可回载、可删除"],
    ["左侧-局域网分享", "一键开放给手机/其他电脑（强制 API Key）"],
    ["左侧-运维", "重启引擎、更新引擎、导出诊断包、卸载"],
    ["右侧-对话", "流式聊天；每条回复显示实测速度（tok/s）；系统提示词、温度、最大输出可调；导出 Markdown"],
], [3.2 * cm, 11.3 * cm]))
story.append(Paragraph("表 2：控制台功能一览", S["caption"]))
story.append(Paragraph("3.3 模型宿主（多模型管理）", S["h2"]))
story.append(Preformatted("python -m modelforge hub        # http://127.0.0.1:18092", S["code"]))
story.append(Paragraph(
    "一个页面管理本机所有已安装模型：自动扫描实例，卡片显示状态/档位/显卡，"
    "一键启动/停止/直达控制台。", S["body"]))
story.append(PageBreak())

# ---- 4 API ----
story.append(Paragraph("4. 接入你的应用（API）", S["h1"]))
story.append(Paragraph("OpenAI SDK（改两行）：", S["body"]))
story.append(Preformatted(
    'from openai import OpenAI\n'
    'client = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="任意值")\n'
    'resp = client.chat.completions.create(\n'
    '    model="任意名", messages=[{"role": "user", "content": "你好"}])\n'
    'print(resp.choices[0].message.content)', S["code"]))
story.append(Paragraph("Anthropic 协议（Claude Code 等）：", S["body"]))
story.append(Preformatted('set ANTHROPIC_BASE_URL=http://127.0.0.1:8081   # 指向 /v1/messages',
                          S["code"]))
story.append(Paragraph("MCP 服务器（编程助手直接管理本地模型）：", S["body"]))
story.append(Preformatted(
    '{\n  "mcpServers": {\n    "modelforge": {\n'
    '      "command": "python",\n'
    '      "args": ["<安装包目录>/mcp_server.py", "--config", "<安装目录>/state/config.json"]\n'
    "    }\n  }\n}", S["code"]))
story.append(Paragraph(
    "提供 5 个工具：model_status / model_metrics / model_info / model_chat / model_restart。",
    S["body"]))

# ---- 5 运维与 FAQ ----
story.append(Paragraph("5. 运维与常见问题", S["h1"]))
story.append(table([
    ["操作", "方法"],
    ["开机自启", "安装向导勾选（默认），或控制台/命令行注册，免管理员"],
    ["更新引擎", "控制台「更新引擎」按钮，或 installer.py --update（模型不动）"],
    ["导出诊断包", "控制台按钮（自动脱敏），反馈问题时发给打包者"],
    ["卸载", "控制台「卸载」红色按钮（两次确认，自动清自启项），或 installer.py --uninstall；--also-package 连带删除分发 zip"],
    ["端口被占用", "无需处理，自动顺延（8080→8081…），页面显示真实地址"],
    ["推理很慢", "内存不足走了 SSD 流式——换更小量化档或关闭大内存程序"],
    ["断网中断", "重跑一次，断点续传不重复下载"],
], [3.4 * cm, 11.1 * cm]))
story.append(Paragraph("表 3：运维与 FAQ", S["caption"]))
story.append(Spacer(1, 10))
story.append(Paragraph(
    "安全说明：默认仅监听本机 127.0.0.1；开放局域网必须设 API Key（否则拒绝启动）；"
    "安装包支持 GPG 签名（篡改拒绝安装）；模型文件全部 SHA256 校验；零遥测。", S["body"]))

OUT.parent.mkdir(exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=A4, topMargin=2.5 * cm, bottomMargin=2.2 * cm,
                        leftMargin=2.8 * cm, rightMargin=2.5 * cm,
                        title="ModelForge 使用手册", author="ModelForge")
doc.build(story, onFirstPage=cover, onLaterPages=header_footer)
print("PDF 已生成:", OUT, f"{OUT.stat().st_size / 1024:.0f} KB")
