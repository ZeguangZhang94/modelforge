# Contributing to ModelForge

感谢你的兴趣！ModelForge 用纯 Python 标准库写成，零第三方依赖，上手成本低。

## 开发环境

```bash
git clone https://github.com/ZeguangZhang94/modelforge.git
cd modelforge
python -m modelforge gui          # 打包台（无需安装任何依赖）
```

## 测试

```bash
python tests/test_planner.py       # 资源规划器单测
python tests/test_engine_args.py   # 引擎参数生成单测
python tests/fake_model/make_fake_model.py   # 生成端到端测试假模型
```

## 提交规范

- 代码：纯标准库，不加第三方依赖（这是核心设计约束）
- 运行时骨架（`runtime_bundle/`）必须能在 Python 3.9+ 裸环境运行
- 提交前跑一遍上面的两个单测
- 大改动请先开 Issue 讨论

## 报告问题

开 Issue 时请附上安装器生成的诊断包（控制台「导出诊断包」按钮，已自动脱敏）。
