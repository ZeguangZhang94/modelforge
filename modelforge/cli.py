"""modelforge 命令行入口（纯标准库 argparse）。

用法：
    python -m modelforge analyze  <模型目录>
    python -m modelforge build    <模型目录> --name <安装包名> [--version 1.0.0] [--publisher 名字] [--context 8192] [--out dist]
    python -m modelforge validate <modelpack.json>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyzer import analyze
from .builder import assemble, build_pack, zip_package
from .estimator import estimate_all, print_estimate_table
from .modelpack import ModelPack, ModelPackError


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        profile = analyze(args.model_dir)
        estimates = estimate_all(profile, args.context)
    except (FileNotFoundError, ValueError) as e:
        print(f"分析失败: {e}", file=sys.stderr)
        return 1
    print(print_estimate_table(profile, estimates))
    print(f"\n能力: {', '.join(profile.capabilities)}  引擎后端: {profile.engine_backend}  许可证: {profile.license}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from .builder import bundle_engines
    from .modelpack import Branding
    branding = Branding(
        display_name=args.display_name,
        logo_text=args.logo,
        theme_color=args.theme_color,
        welcome_message=args.welcome,
        about=args.about,
        license_text=args.license_text,
    )
    try:
        pack, profile = build_pack(
            model_dir=args.model_dir,
            name=args.name,
            version=args.version,
            publisher=args.publisher,
            context=args.context,
            engine_tag=args.engine_tag,
            branding=branding,
        )
    except (FileNotFoundError, ValueError, ModelPackError) as e:
        print(f"打包失败: {e}", file=sys.stderr)
        return 1

    print(print_estimate_table(profile, estimate_all(profile, args.context)))
    print(f"\n开始组装安装包 {pack.name}-{pack.version} ...")
    try:
        root = assemble(pack, args.model_dir, args.out)
        if args.bundle_engines:
            n = bundle_engines(root, args.bundle_engines)
            print(f"已内置引擎文件 {n} 个（离线可装）")
        if args.bundle_python:
            from .builder import bundle_python
            bundle_python(root, args.bundle_python)
            print("已内置 Python 运行时（终端零依赖）")
        if args.sign:
            from .signing import sign_package
            sig = sign_package(root)
            print(f"已签名: {sig.name}" if sig else "未找到 gpg 密钥，跳过签名")
        zip_path = zip_package(root)
    except (FileNotFoundError, OSError) as e:
        print(f"组装失败: {e}", file=sys.stderr)
        return 1

    print(f"包目录: {root}")
    print(f"分发文件: {zip_path}")
    print("\n终端用户使用方式：解压后 Windows 双击 START-HERE.bat，Linux/macOS 运行 ./install.sh")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        pack = ModelPack.load(args.modelpack)
    except ModelPackError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"modelpack 校验通过: {pack.name} {pack.version}（{len(pack.variants)} 个变体）")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import run_gui
    return run_gui(args.port)


def cmd_hub(args: argparse.Namespace) -> int:
    from .hub import run_hub
    return run_hub(args.port)


def cmd_convert(args: argparse.Namespace) -> int:
    from .convert import convert
    return convert(args.model_dir, args.out, args.quant)


def cmd_build_update(args: argparse.Namespace) -> int:
    """生成离线增量更新包（只含变化的运行时文件，不含模型）。"""
    from .builder import build_update_pack
    try:
        zp = build_update_pack(args.old, args.new, args.out)
    except (ValueError, FileNotFoundError) as e:
        print(f"生成更新包失败: {e}", file=sys.stderr)
        return 1
    print(f"更新包: {zp}（{zp.stat().st_size / 1e6:.1f} MB）")
    print("摆渡流程：拷贝到离线机 → 在目标安装包目录运行 installer.py --apply-update <zip>")
    return 0


def cmd_build_dmr(args: argparse.Namespace) -> int:
    """生成 Linux Docker Model Runner 安装包。"""
    from .dmr import build_dmr_package
    ref = args.repo
    if not ref.startswith(("hf.co/", "ai/")) and "/" in ref and not ref.startswith("ai/"):
        ref = "hf.co/" + ref  # 裸 HF 仓库名自动加前缀
    zp = build_dmr_package(args.name, ref, args.out, args.display_name)
    print(f"DMR 安装包: {zp}")
    print("终端（Linux）: 解压后运行 ./install.sh（自动检测 docker、拉模型、给出 API 地址）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="modelforge",
        description="AI 模型本地安装包生成框架：选模型、输名称，一键产出可分发安装包。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="分析模型目录，输出硬件需求预估表")
    p_an.add_argument("model_dir", type=Path)
    p_an.add_argument("--context", type=int, default=8192)
    p_an.set_defaults(func=cmd_analyze)

    p_b = sub.add_parser("build", help="生成模型安装包（目录 + zip）")
    p_b.add_argument("model_dir", type=Path)
    p_b.add_argument("--name", required=True, help="安装包名称（小写字母/数字/._-）")
    p_b.add_argument("--version", default="1.0.0")
    p_b.add_argument("--publisher", default="")
    p_b.add_argument("--context", type=int, default=8192)
    p_b.add_argument("--engine-tag", default="",
                     help="锁定 llama.cpp release 版本（如 b6971），安装时自动下载引擎")
    p_b.add_argument("--bundle-engines", default="", metavar="目录",
                     help="把本地引擎二进制目录打进安装包（离线可装）")
    p_b.add_argument("--bundle-python", default="", metavar="embed.zip",
                     help="把 Python embeddable 打进安装包（终端零依赖，Windows）")
    p_b.add_argument("--sign", action="store_true",
                     help="用 GPG 对安装包签名（需要本机已配置 gpg 密钥）")
    p_b.add_argument("--display-name", default="", help="界面显示名")
    p_b.add_argument("--logo", default="⚡", help="标题旁 Logo（emoji 或短文本）")
    p_b.add_argument("--theme-color", default="#4c9aff", help="主题色（hex）")
    p_b.add_argument("--welcome", default="", help="控制台欢迎语")
    p_b.add_argument("--about", default="", help="关于说明文字")
    p_b.add_argument("--license-text", default="", help="自定义许可证文案")
    p_b.add_argument("--out", type=Path, default=Path("dist"))
    p_b.set_defaults(func=cmd_build)

    p_v = sub.add_parser("validate", help="校验 modelpack.json")
    p_v.add_argument("modelpack", type=Path)
    p_v.set_defaults(func=cmd_validate)

    p_g = sub.add_parser("gui", help="启动打包者 Web 界面（打包台）")
    p_g.add_argument("--port", type=int, default=18090)
    p_g.set_defaults(func=cmd_gui)

    p_d = sub.add_parser("build-dmr", help="生成 Linux Docker Model Runner 安装包")
    p_d.add_argument("--name", required=True)
    p_d.add_argument("--repo", required=True,
                     help="模型引用：HF 仓库名（自动加 hf.co/ 前缀）或 ai/xxx Hub 官方名")
    p_d.add_argument("--display-name", default="")
    p_d.add_argument("--out", type=Path, default=Path("dist"))
    p_d.set_defaults(func=cmd_build_dmr)

    p_h = sub.add_parser("hub", help="多模型宿主控制台（管理所有已安装模型）")
    p_h.add_argument("--port", type=int, default=18092)
    p_h.set_defaults(func=cmd_hub)

    p_c = sub.add_parser("convert", help="safetensors → GGUF 转换（条件不满足时给出引导）")
    p_c.add_argument("model_dir", type=Path)
    p_c.add_argument("--out", type=Path, default=None)
    p_c.add_argument("--quant", default="", help="转换后再量化（如 q4_k_m）")
    p_c.set_defaults(func=cmd_convert)

    p_u = sub.add_parser("build-update", help="生成离线增量更新包（对比两个版本包目录）")
    p_u.add_argument("--old", type=Path, required=True, help="旧版本包目录")
    p_u.add_argument("--new", type=Path, required=True, help="新版本包目录")
    p_u.add_argument("--out", type=Path, default=Path("dist"))
    p_u.set_defaults(func=cmd_build_update)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
