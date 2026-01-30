from __future__ import annotations

import json
from typing import Any, Dict, List

from .types import Constraints, Conflict


def build_prompt(requirement_text: str, constraints: Constraints, conflict_summary: List[Conflict]) -> str:
    """
    组装给 LLM 的提示词：需求文本 + 当前约束 + 上一轮冲突摘要。
    要求：严格输出 PlantUML（只输出代码块），并对 include/extend 关系附带因果说明（用注释行）。
    """
    constraints.normalize_inplace()
    summary = []
    for c in conflict_summary[:20]:
        summary.append({"type": c.conflict_type, "a": c.a, "b": c.b, "details": c.details})

    payload: Dict[str, Any] = {"constraints": constraints.model_dump(), "conflicts": summary}

    return f"""你是 UML 用例图建模专家。请基于【需求文本】生成一份 UML 用例图的 PlantUML 代码。

硬性要求：
1) 只输出一个 ```plantuml 代码块```，不要输出任何额外解释文本。
2) 必须包含 actor、usecase、以及 actor->usecase 的关联。
3) 对于用例之间的 include/extend：
   - include：A 执行过程中必须包含 B（强制性因果依赖）
   - extend：B 在特定条件触发时对 A 扩展（必须写明触发条件/扩展点）
4) 每条 include/extend 关系前，请用注释行（以 ' 开头）写一句简短因果说明/触发条件，不得破坏 PlantUML 语法。
5) 必须遵守【约束】；若约束与需求冲突，以约束为准（视为用户确认）。

【需求文本】
{requirement_text}

【约束与冲突摘要（JSON）】
{json.dumps(payload, ensure_ascii=False, indent=2)}

请输出 PlantUML：
"""
