# ModelForge · 模型熔炉

**把任何一个开源 AI 模型，变成双击即装的本地 AI 服务安装包。**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![Zero deps](https://img.shields.io/badge/dependencies-zero-brightgreen)]()
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20%7C%20AMD%20%7C%20Intel-orange)]()

打包者在「打包台」里选一个模型、填一个名称，框架自动生成该模型专属的安装包；
终端用户拿到安装包后**双击即用**——自动识别显卡、自动匹配量化档位、自动充分利用显存，
装完直接得到 OpenAI / Anthropic 兼容的本地 API 和一个全中文网页控制台。

---

## 为什么做这个项目

Ollama、LM Studio 解决的是"我自己装一个模型来用"。
ModelForge 解决的是另一件事——**把模型打包成产品，分发给完全不懂技术的人**：

- IT 部门打好包发给全体员工，离线可装、带公司品牌
- 开发者把"模型 + 服务"作为交付物发给客户，对方不需要知道 llama.cpp 是什么
- 内容创作者把微调模型打包发给粉丝，双击就出网页控制台
- 隔离网/涉密环境：权重 + 引擎 + Python 运行时全部内置，零出站请求

## 核心特性

**🎯 对终端用户极致简单**
- 双击 `START-HERE.bat` → 浏览器安装向导 → 四次点击完成
- 内置 Python embeddable 运行时，**连 Python 都不用装**
- 端口被占用自动顺延，断网续传，中断重跑不重复下载

**⚡ 对硬件极致利用**
- 自动探测 NVIDIA / AMD / Intel 显卡并评分选卡，多卡自动聚合切分
- AMD Windows 显存走注册表 64 位精确读取（绕开 WMI 4GB 溢出 bug）
- 显存/内存/SSD 三级资源规划：MoE 专家卸载、稠密层卸载、KV 缓存量化自动降级
- **MTP / EAGLE3 投机解码自动检测接线**（扫描 GGUF 张量名，无需配置）

**🧩 对打包者极致灵活**
- 模型商店：17 个精选模型卡片（Qwen / Llama / DeepSeek / Coder / VL），点击即下载
- 品牌自定义：显示名 / Logo / 主题色 / 欢迎语 / 许可证文案，向导与控制台全部生效
- GPG 签名：安装器验证失败拒绝安装
- 一个 ModelPack 清单定义一切，安装器与模型完全解耦

**🏢 企业级就绪**
- 静默安装契约：`START-HERE.bat /S /DIR=... /NOSTART`，返回码 0/1603，幂等
- 自带 `deploy/` 套件：Intune 检测脚本 + SCCM/GPO 部署指南
- RBAC 四角色（consumer/operator/auditor/admin）+ 多虚拟密钥 + 按密钥用量核算
- 审计日志 JSONL（时间/身份/模型版本/权重指纹/摘要 hash），Prometheus 指标接口
- 离线增量更新包（air-gap 摆渡流程）

**🔌 开放兼容**
- OpenAI 兼容 `/v1/chat/completions` + Anthropic 兼容 `/v1/messages`
- MCP 服务器：Claude Code 等编程助手可直接管理本地模型
- RAG 文档问答（纯标准库 TF-IDF，零依赖）
- 控制台全中文：实测速度显示、历史持久化、局域网一键分享（强制 API Key）

## 实测性能

AMD RX 7900 XTX 24GB + 68GB RAM，Windows 11，Vulkan 后端：

| 模型 | 档位 | 输出速度 |
|---|---|---|
| Qwen2.5 0.5B | Q4_K_M | 259 tokens/s |
| Qwen2.5 1.5B | Q4_K_M | 327 tokens/s |
| Qwen3.8 27B | UD-Q4_K_M + MTP 投机解码 | 43.7 tokens/s |

> 27B 的 41~45 tok/s 已达该卡显存带宽理论上限的 70%+；实测三个量化档位中 Q4_K_M 反而是最快的（IQ 系反量化开销反噬）。

## 30 秒上手

**终端用户**（拿到安装包的人）：

```
Windows: 解压 zip → 双击 START-HERE.bat → 浏览器向导点四次 → 开始聊天
Linux:   解压 → ./install.sh
```

**打包者**（做安装包的人）：

```bash
git clone https://github.com/ZeguangZhang94/modelforge.git
cd modelforge
python -m modelforge gui        # 打开打包台（零依赖，无需 pip install）
```

打包台里：模型商店选模型 → 一键下载 → 填名称 → 生成安装包 → 一键安装启动。
纯 Python 标准库实现，克隆即用。

**其他入口**：

```bash
python -m modelforge hub        # 模型宿主：一个页面管理所有已安装模型
python -m modelforge build-dmr --name xxx --repo org/model-GGUF   # Linux 走 Docker Model Runner
```

## 文档

| 文档 | 内容 |
|---|---|
| [架构设计](docs/02-架构设计.md) | ModelPack 契约、选卡算法、三级资源规划 |
| [CONTRIBUTING](CONTRIBUTING.md) | 贡献指南（核心约束：纯标准库零依赖） |

> 使用手册 PDF 随安装包分发（打包者可用 `docs/build_manual_pdf.py` 自行生成定制版）。

## 设计致谢

推理引擎基于 [llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT）。
一键安装与三级显存分布的设计思想借鉴自 [Strata](https://github.com/Niko1221/Strata)（MIT）。
安装包内分发的模型权重各有其许可证，分发前请自行确认。

## License

[MIT](LICENSE) © 2026 ModelForge Contributors
