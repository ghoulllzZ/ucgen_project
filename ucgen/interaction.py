from __future__ import annotations

import logging
from typing import Any, Dict, List

import typer

from .merge import normalize_name
from .types import Conflict, Constraints

logger = logging.getLogger("ucgen.interaction")


def generate_questions(unresolved: List[Conflict]) -> List[Dict[str, Any]]:
    """对 unresolved 生成低认知负担问题（是/否/单选/补条件）。"""
    questions: List[Dict[str, Any]] = []
    qid = 1

    for c in unresolved:
        if c.conflict_type == "edge_kind_mismatch":
            a, b = c.a, c.b
            kinds = c.details.get("kinds", [])
            questions.append(
                {
                    "id": f"q{qid}",
                    "type": "single_choice",
                    "target": {"a": a, "b": b},
                    "prompt": f"在业务上，【{a}】与【{b}】更符合哪种关系？",
                    "choices": [
                        {"value": "include", "text": f"{a} 必须包含 {b}（include：必经子步骤）"},
                        {"value": "extend", "text": f"{a} 在特定条件下扩展 {b}（extend：条件触发，需条件/扩展点）"},
                        {"value": "none", "text": "两者之间不需要 include/extend 关系"},
                    ],
                    "observed": kinds,
                }
            )
            qid += 1

        elif c.conflict_type == "edge_presence_uncertain":
            a, b = c.a, c.b
            questions.append(
                {
                    "id": f"q{qid}",
                    "type": "yes_no_include",
                    "target": {"base": a, "included": b},
                    "prompt": f"用例【{a}】执行过程中是否【必须】包含用例【{b}】？（是=include，否=不建立 include）",
                    "default": "yes" if (normalize_name(b) == normalize_name("选择图形") and normalize_name(a) != normalize_name("创建图形")) else "no",
                }
            )
            qid += 1

        elif c.conflict_type == "extend_missing_condition":
            ext, base = c.a, c.b
            questions.append(
                {
                    "id": f"q{qid}",
                    "type": "extend_condition",
                    "target": {"extension": ext, "base": base},
                    "prompt": f"如果【{ext}】是对【{base}】的 extend，请补充触发条件/扩展点（若不需要该 extend，请输入 NONE）",
                    "default": "按住Shift时" if (normalize_name(ext).find("等比例") >= 0 or "Shift" in c.details.get("msg", "")) else "NONE",
                }
            )
            qid += 1

        elif c.conflict_type == "actor_uncertain":
            a = c.a
            default = "yes" if float(c.details.get("freq", 0.0)) >= 0.6 else "no"
            questions.append(
                {
                    "id": f"q{qid}",
                    "type": "actor_yes_no",
                    "target": {"actor": a},
                    "prompt": f"参与者【{a}】是否确实存在并需要保留在用例图中？（是=保留，否=移除）",
                    "default": default,
                }
            )
            qid += 1

        elif c.conflict_type == "usecase_uncertain":
            u = c.a
            default = "yes" if float(c.details.get("freq", 0.0)) >= 0.6 else "no"
            questions.append(
                {
                    "id": f"q{qid}",
                    "type": "usecase_yes_no",
                    "target": {"usecase": u},
                    "prompt": f"用例【{u}】是否确实存在并需要保留在最终用例图中？（是=保留，否=移除）",
                    "default": default,
                }
            )
            qid += 1

        elif c.conflict_type == "relation_unknown":
            a, b = c.a, c.b
            kind = c.details.get("kind", "include")
            if kind == "include":
                questions.append(
                    {
                        "id": f"q{qid}",
                        "type": "yes_no_include",
                        "target": {"base": a, "included": b},
                        "prompt": f"是否确认：用例【{a}】执行时必须包含【{b}】？（是=include，否=不建立 include）",
                        "default": "no",
                    }
                )
            else:
                questions.append(
                    {
                        "id": f"q{qid}",
                        "type": "extend_condition",
                        "target": {"extension": a, "base": b},
                        "prompt": f"如果【{a}】是对【{b}】的 extend，请补充触发条件/扩展点（若不需要该 extend，请输入 NONE）",
                        "default": "NONE",
                    }
                )
            qid += 1

    return questions


def apply_answers_to_constraints(constraints: Constraints, questions: List[Dict[str, Any]], answers: Dict[str, Any]) -> Constraints:
    """将用户回答转为结构化约束，并与现有约束合并。"""
    c = constraints.model_copy(deep=True)
    c.normalize_inplace()

    for q in questions:
        qid = q["id"]
        ans = answers.get(qid)
        if ans is None:
            continue

        if q["type"] == "yes_no_include":
            base = q["target"]["base"]
            inc = q["target"]["included"]
            if str(ans).lower() in ("y", "yes", "true", "是", "1"):
                c.must_include.append({"base": base, "included": inc})
            else:
                c.forbid_include.append({"base": base, "included": inc})

        if q["type"] == "single_choice":
            a = q["target"]["a"]
            b = q["target"]["b"]
            if ans == "include":
                c.must_include.append({"base": a, "included": b})
                c.forbid_extend.append({"base": b, "extension": a})
            elif ans == "extend":
                c.extend.append({"base": b, "extension": a, "condition": "", "extension_point": ""})
                c.forbid_include.append({"base": a, "included": b})
            else:
                c.forbid_include.append({"base": a, "included": b})
                c.forbid_extend.append({"base": b, "extension": a})

        if q["type"] == "extend_condition":
            ext = q["target"]["extension"]
            base = q["target"]["base"]
            if isinstance(ans, str) and ans.strip().upper() == "NONE":
                c.forbid_extend.append({"base": base, "extension": ext})
            else:
                # 允许用户写一段文字，我们作为 condition 写入；extension_point 可后续再细分
                c.extend.append({"base": base, "extension": ext, "condition": str(ans).strip(), "extension_point": ""})

        if q["type"] == "actor_yes_no":
            a = q["target"]["actor"]
            if str(ans).lower() in ("y", "yes", "true", "是", "1"):
                c.keep_actors.append(a)
            else:
                c.forbid_actors.append(a)

        if q["type"] == "usecase_yes_no":
            u = q["target"]["usecase"]
            if str(ans).lower() in ("y", "yes", "true", "是", "1"):
                c.keep_usecases.append(u)
            else:
                c.forbid_usecases.append(u)

    # 去重
    def _dedup_list(lst: List[str]) -> List[str]:
        out = []
        seen = set()
        for x in lst:
            k = normalize_name(x)
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    def _dedup_dict(lst: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        out = []
        for x in lst:
            key = tuple(sorted(x.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append(x)
        return out

    c.must_include = _dedup_dict(c.must_include)
    c.forbid_include = _dedup_dict(c.forbid_include)
    c.extend = _dedup_dict(c.extend)
    c.forbid_extend = _dedup_dict(c.forbid_extend)
    c.keep_actors = _dedup_list(c.keep_actors)
    c.forbid_actors = _dedup_list(c.forbid_actors)
    c.keep_usecases = _dedup_list(c.keep_usecases)
    c.forbid_usecases = _dedup_list(c.forbid_usecases)

    c.normalize_inplace()
    return c


def collect_answers(questions: List[Dict[str, Any]], interactive: bool) -> Dict[str, Any]:
    """交互或自动（--no-interactive）收集回答。"""
    answers: Dict[str, Any] = {}
    for q in questions:
        qid = q["id"]

        if not interactive:
            answers[qid] = q.get("default", "no")
            continue

        typer.echo("")
        typer.echo(q["prompt"])

        default = q.get("default", "")
        if q["type"] == "single_choice":
            for ch in q["choices"]:
                typer.echo(f"- {ch['value']}: {ch['text']}")
            answers[qid] = typer.prompt("请输入选项值", default=q["choices"][0]["value"])
        elif q["type"] in ("yes_no_include", "actor_yes_no", "usecase_yes_no"):
            answers[qid] = typer.prompt("请输入 是/否", default=default or "no")
        elif q["type"] == "extend_condition":
            answers[qid] = typer.prompt("请输入条件（或 NONE）", default=default or "NONE")
        else:
            answers[qid] = typer.prompt("请输入", default=default)

    return answers
