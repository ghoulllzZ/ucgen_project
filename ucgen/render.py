from __future__ import annotations
from .plantuml import normalize_usecase_plantuml  # [PATCH]


import logging
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger("ucgen.render")

RenderMode = Literal["auto", "off", "png", "svg"]


def _has_plantuml_cli() -> bool:
    return shutil.which("plantuml") is not None


def render_plantuml(puml_path: Path, mode: RenderMode = "auto") -> Optional[Path]:
    """
    尝试调用本机 plantuml CLI 渲染 png/svg。
    缺失 plantuml/java 环境时：跳过且不报错。
    """
    if mode == "off":
        return None

    if not _has_plantuml_cli():
        logger.info("plantuml CLI not found; skip rendering.")
        return None

    out_dir = puml_path.parent
    fmt = "png" if mode in ("auto", "png") else "svg"

    try:
        # [PATCH] 渲染前自动修正候选 PlantUML，避免 PlantUML 语法错误导致无法出图
        raw = puml_path.read_text(encoding="utf-8", errors="ignore")
        fixed, fixes = normalize_usecase_plantuml(raw)
        if fixed != raw:
            puml_path.write_text(fixed, encoding="utf-8")
            logger.warning("PlantUML normalized before rendering (%s): %s", puml_path.name, "; ".join(fixes)[:300])
        subprocess.run(
            ["plantuml", f"-t{fmt}", str(puml_path)],
            cwd=str(out_dir),
            check=False,
            capture_output=True,
            text=True,
        )
        out_file = puml_path.with_suffix(f".{fmt}")
        if out_file.exists():
            logger.info("Rendered: %s", out_file)
            return out_file
        logger.warning("Render command executed, but output not found.")
        return None
    except Exception as e:
        logger.warning("Render skipped due to error: %s", e)
        return None
