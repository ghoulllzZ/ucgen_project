from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Set

from .merge import MergedModel, normalize_name
from .types import Conflict, Constraints, DiagramIR, EdgeKey, Relation

logger = logging.getLogger("ucgen.reasoning")


@dataclass
class RuleDecision:
    kept_edges: Set[EdgeKey]
    dropped_edges: Set[EdgeKey]
    unresolved_conflicts: List[Conflict]
    notes: Dict[str, str]


def _has_condition_text(meta_list: List[Dict]) -> bool:
    for m in meta_list:
        cond = (m.get("condition") or "").strip()
        label = (m.get("label") or "").lower()
        if cond:
            return True
        if "当" in label or "如果" in label or "条件" in label or "condition" in label:
            return True
    return False


def _req_mentions(requirement: str, a: str, b: str) -> bool:
    ra = normalize_name(a)
    rb = normalize_name(b)
    req = normalize_name(requirement)
    return (ra in req) and (rb in req)


def _default_include_heuristic(base: str, included: str) -> bool:
    b = normalize_name(base)
    i = normalize_name(included)
    # 领域先验：对“移动/删除/调整尺寸”操作通常需要先选择目标图形
    if i == normalize_name("选择图形") and any(
        x in b
        for x in [
            normalize_name("移动图形"),
            normalize_name("删除图形"),
            normalize_name("调整图形尺寸"),
            normalize_name("改变图形尺寸"),
            normalize_name("调整尺寸"),
        ]
    ):
        return True
    return False


class PythonRuleEngine:
    """默认可用的内置规则/推理模块（无需外部依赖）。

    规则优先级：
    1) 用户约束（Constraints）
    2) 关系语义约束：
       - include：倾向于“必经子步骤”
       - extend：必须具备触发条件/扩展点
    3) 需求文本显式证据与轻量先验
    """

    def decide(self, merged: MergedModel, requirement: str, constraints: Constraints) -> RuleDecision:
        constraints.normalize_inplace()

        kept: Set[EdgeKey] = set()
        dropped: Set[EdgeKey] = set()
        unresolved: List[Conflict] = []
        notes: Dict[str, str] = {}

        forbid_include = {(normalize_name(x["base"]), normalize_name(x["included"])) for x in constraints.forbid_include}
        forbid_extend = {(normalize_name(x["extension"]), normalize_name(x["base"])) for x in constraints.forbid_extend}

        for ek in merged.edge_stats.keys():
            if ek.kind == "include" and (ek.src, ek.dst) in forbid_include:
                dropped.add(ek)
                notes[f"drop:{ek}"] = "forbid_include"
            if ek.kind == "extend" and (ek.src, ek.dst) in forbid_extend:
                dropped.add(ek)
                notes[f"drop:{ek}"] = "forbid_extend"

        must_include = {(normalize_name(x["base"]), normalize_name(x["included"])) for x in constraints.must_include}
        must_extend = {(normalize_name(x["extension"]), normalize_name(x["base"])) for x in constraints.extend}
        extend_condition = {(normalize_name(x["extension"]), normalize_name(x["base"])): x.get("condition", "") for x in constraints.extend}

        for (base, inc) in must_include:
            kept.add(EdgeKey(src=base, kind="include", dst=inc))
            notes[f"keep:include:{base}->{inc}"] = "must_include"

        for (ext, base) in must_extend:
            kept.add(EdgeKey(src=ext, kind="extend", dst=base))
            notes[f"keep:extend:{ext}->{base}"] = f"must_extend condition={extend_condition.get((ext, base), '')}".strip()

        for ek, es in merged.edge_stats.items():
            if ek in dropped or ek in kept:
                continue

            if ek.kind in ("assoc", "generalize"):
                kept.add(ek)
                continue

            if ek.kind == "include":
                # 对同一对节点存在 include/extend 冲突：暂时交给澄清
                if any(c.conflict_type == "edge_kind_mismatch" and c.a == ek.src and c.b == ek.dst for c in merged.conflicts):
                    continue

                # 低置信关系：用启发式/文本证据决定
                if any(c.conflict_type == "edge_presence_uncertain" and c.a == ek.src and c.b == ek.dst for c in merged.conflicts):
                    if _default_include_heuristic(ek.src, ek.dst) or _req_mentions(requirement, ek.src, ek.dst):
                        kept.add(ek)
                        notes[f"keep:include:{ek.src}->{ek.dst}"] = "heuristic_or_mention"
                    else:
                        dropped.add(ek)
                        notes[f"drop:include:{ek.src}->{ek.dst}"] = "low_confidence"
                else:
                    kept.add(ek)

            if ek.kind == "extend":
                # extend 必须有触发条件
                has_cond = _has_condition_text(es.meta) or bool(extend_condition.get((ek.src, ek.dst), "").strip())
                if not has_cond:
                    unresolved.append(
                        Conflict(
                            conflict_type="extend_missing_condition",
                            a=ek.src,
                            b=ek.dst,
                            details={"msg": "extend 缺少触发条件/扩展点，需用户补充或删除该关系"},
                        )
                    )
                    continue
                kept.add(ek)

        # 冲突（include vs extend）仍未解决：保留为 unresolved 以便澄清
        for c in merged.conflicts:
            if c.conflict_type == "edge_kind_mismatch":
                include_key = EdgeKey(src=c.a, kind="include", dst=c.b)
                extend_key = EdgeKey(src=c.a, kind="extend", dst=c.b)
                if include_key not in kept and extend_key not in kept:
                    unresolved.append(c)

        return RuleDecision(kept_edges=kept, dropped_edges=dropped, unresolved_conflicts=unresolved, notes=notes)


def clingo_available() -> bool:
    try:
        import clingo  # noqa: F401
        return True
    except Exception:
        return False


def check_with_clingo_stub(constraints: Constraints) -> Dict[str, str]:
    """
    clingo 增强（可选）：此处提供一致性检查的可运行 stub。
    - 若安装 clingo，则返回 engine=clingo，并对 must/forbid 自相矛盾做提示。
    - 未安装则 engine=python_stub。
    """
    constraints.normalize_inplace()
    must_i = {(x["base"], x["included"]) for x in constraints.must_include}
    forb_i = {(x["base"], x["included"]) for x in constraints.forbid_include}
    must_e = {(x["extension"], x["base"]) for x in constraints.extend}
    forb_e = {(x["extension"], x["base"]) for x in constraints.forbid_extend}

    res: Dict[str, str] = {"engine": "clingo" if clingo_available() else "python_stub"}
    bad_i = must_i & forb_i
    bad_e = must_e & forb_e
    if bad_i:
        res["inconsistent_include"] = json.dumps(sorted(list(bad_i)), ensure_ascii=False)
    if bad_e:
        res["inconsistent_extend"] = json.dumps(sorted(list(bad_e)), ensure_ascii=False)
    res["status"] = "ok" if not bad_i and not bad_e else "inconsistent"
    return res


def build_final_diagram_from_kept(merged: MergedModel, kept: Set[EdgeKey]) -> DiagramIR:
    """从合并图 + kept 边集合构造最终 IR。"""
    actors: Set[str] = set()
    usecases: Set[str] = set()
    relations: List[Relation] = []

    incident: Set[str] = set()
    for ek in kept:
        incident.add(ek.src)
        incident.add(ek.dst)

    for n, st in merged.node_stats.items():
        if st.status == "common" or n in incident:
            if st.node_type == "actor":
                actors.add(n)
            else:
                usecases.add(n)

    for ek in kept:
        if ek.src not in actors and ek.src not in usecases:
            continue
        if ek.dst not in actors and ek.dst not in usecases:
            continue
        relations.append(Relation(src=ek.src, kind=ek.kind, dst=ek.dst, meta={"source": "final"}))

    return DiagramIR(actors=actors, usecases=usecases, relations=relations, alias_to_name={})
