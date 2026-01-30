# 交互式 UML 用例图自动生成系统（ucgen）

本项目实现一个“交互式 UML 用例图自动生成系统”，输入中文自然语言需求，经过：
- 多模型/多采样候选生成（默认 MockLLM，无需任何商业 API Key 也可跑通）
- PlantUML 解析 → 对齐合并统一表示 → 冲突检测
- 基于因果逻辑约束的规则判定（默认内置规则引擎；若本机安装 clingo 则自动增强一致性检查）
- 人机交互澄清（可交互/可无交互自动答复）
- 多轮迭代收敛
最终输出 PlantUML（final_usecase.puml）与全流程日志报告（report.json），并在检测到本机 plantuml/java 环境时可选渲染 png/svg（缺失则跳过且不报错）。

---

## 运行环境
- Python >= 3.10

---

## 安装
进入项目根目录后执行：

```bash
pip install -r requirements.txt
```

---

## 快速开始（默认 MockLLM，可直接跑通）
```bash
python -m ucgen run --req examples/req_graph_editor.txt --out out --max-iters 5 --backend mock --no-interactive
```

运行后将生成：
- `out/final_usecase.puml`
- `out/report.json`
- 每轮迭代目录：`out/iter_1/ ... out/iter_k/`
  - `candidates/*.puml`
  - `merged_graph.json`
  - `questions.json`
  - `answers.json`
  - `rule_report.json`

---

## CLI 使用说明

### 主命令
```bash
python -m ucgen run --req <需求文件> --out <输出目录> [其他参数...]
```

### 常用参数
- `--backend {mock,openai,deepseek,qwen,multi}`：后端选择（默认 mock）
- `--temperature FLOAT`：采样温度（默认 0.9，建议 0.7~1.0）
- `--n-samples INT`：每模型采样次数（默认 6，建议 5~10）
- `--models TEXT`：multi 模式下模型列表（可多次传入），例如：`--models gpt-4 --models deepseek-r1 --models qwen3`
- `--seed INT`：随机种子
- `--interactive/--no-interactive`：是否交互澄清（默认 interactive）
- `--jaccard-threshold FLOAT`：收敛阈值（默认 0.98）
- `--render {auto,off,png,svg}`：渲染策略（默认 auto）
- `--config PATH`：可选 config.yaml（示例见 `config.yaml.example`）

---

## 配置真实 LLM（可选，未配置则自动 fallback 到 MockLLM）

你可以通过环境变量或 `config.yaml`（示例见 `config.yaml.example`）配置：

### OpenAI（骨架）
```bash
set OPENAI_API_KEY=YOUR_KEY
set OPENAI_BASE_URL=https://api.openai.com/v1
set OPENAI_MODEL=gpt-4o-mini
```

### DeepSeek（骨架）
```bash
set DEEPSEEK_API_KEY=YOUR_KEY
set DEEPSEEK_BASE_URL=https://api.deepseek.com
set DEEPSEEK_MODEL=deepseek-chat
```

### Qwen（骨架）
```bash
set QWEN_API_KEY=YOUR_KEY
set QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
set QWEN_MODEL=qwen-plus
```

> 注意：本项目对真实 API 调用仅提供“工程化适配器骨架 + 错误提示 + 自动 fallback mock”，保证无 Key 时流程也能完整跑通。

---

## 单元测试

```bash
pytest -q
```

---
