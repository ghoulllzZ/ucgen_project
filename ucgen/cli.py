from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import typer

from .logging_utils import setup_logging
from .pipeline import run_pipeline
from .types import RunArgs

app = typer.Typer(add_completion=False, help="交互式 UML 用例图自动生成系统（ucgen）")


@app.command("run")
def run(
    req: Path = typer.Option(..., "--req", exists=True, file_okay=True, dir_okay=False, help="需求文本文件路径"),
    out: Path = typer.Option(Path("out"), "--out", help="输出目录"),
    max_iters: int = typer.Option(4, "--max-iters", min=1, max=10, help="最大迭代轮次（默认 4）"),
    backend: str = typer.Option("mock", "--backend", help="后端：mock/openai/deepseek/qwen/gemini/multi"),
    temperature: float = typer.Option(0.9, "--temperature", min=0.0, max=2.0, help="采样温度（建议 0.7~1.0）"),
    n_samples: int = typer.Option(6, "--n-samples", min=1, max=20, help="每模型采样次数（建议 5~10）"),
    models: List[str] = typer.Option([], "--models", help="multi 模式下模型列表（可多次传入）"),
    seed: int = typer.Option(7, "--seed", help="随机种子"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="是否进行交互式澄清"),
    jaccard_threshold: float = typer.Option(0.98, "--jaccard-threshold", min=0.0, max=1.0, help="收敛阈值"),
    render: str = typer.Option("auto", "--render", help="渲染：auto/off/png/svg"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True, file_okay=True, dir_okay=False, help="可选 config.yaml"),
    log_level: str = typer.Option("INFO", "--log-level", help="日志级别：DEBUG/INFO/WARNING/ERROR"),
):
    setup_logging(level=log_level)
    logging.getLogger("ucgen").info("Starting ucgen run...")

    args = RunArgs(
        req_path=req,
        out_dir=out,
        max_iters=max_iters,
        backend=backend,
        temperature=temperature,
        n_samples=n_samples,
        models=models,
        seed=seed,
        interactive=interactive,
        jaccard_threshold=jaccard_threshold,
        render=render,
        config_path=config,
    )
    run_pipeline(args)


if __name__ == "__main__":
    app()
