# ucgen/validator.py
from __future__ import annotations

from typing import Dict, List, Tuple, Set

from .asp_reasoner import extract_cond_extpt_from_justification, asp_check_extend_causality

from .merge import MergedModel, normalize_name
from .types import (
    Conflict,
    Constraints,
    ValidationReport,
    ValidatedActor,
    ValidatedRelation,
    ValidatedUseCase,
)

from .asp_reasoner import asp_prune_usecases


def _mentioned(req: str, phrase: str) -> bool:
    r = normalize_name(req)
    p = normalize_name(phrase)
    return bool(p) and (p in r)


def _include_heuristic(base: str, included: str) -> Tuple[bool, str]:
    """
    轻量 include 启发式：给 Stage3 一个默认判断，避免全 unknown。
    你可以继续扩展。
    """
    b = normalize_name(base)
    i = normalize_name(included)

    # “创建图形”通常必须包含“选择图形类型”
    if ("创建" in b and "图形" in b) and ("选择" in i and ("类型" in i or "图形类型" in i)):
        return True, "领域先验：创建图形通常必须先选择图形类型"
    # “移动/删除/调整尺寸”通常必须包含“选择图形”
    if any(k in b for k in ["移动", "删除", "调整", "改变", "缩放"]) and ("选择" in i and "图形" in i):
        return True, "领域先验：移动/删除/调整尺寸通常必须先选择目标图形"

    return False, ""


def validate_all(req_text: str, merged: MergedModel, constraints: Constraints) -> Tuple[ValidationReport, List[Conflict]]:
    """
    阶段3：对 actor/usecase/relations 做语义验证（默认 Python 规则引擎 + 可选 ASP 增强）。

    ASP 目标：
    - 仅针对“模糊/低共识”的用例，过滤掉“被更抽象用例蕴含的具体用例”
      例：创建图形 + 创建矩形/创建圆形/... => 删除具体用例，仅保留创建图形
    """
    constraints.normalize_inplace()
    total = max(int(getattr(merged, "total_candidates", 1)), 1)

    keep_nodes = set(constraints.keep_actors + constraints.keep_usecases)
    forbid_nodes = set(constraints.forbid_actors + constraints.forbid_usecases)
    keep_usecases = set(constraints.keep_usecases)

    # --------------------
    # [ASP] 计算“模糊用例集合”并做蕴含过滤
    # --------------------
    usecase_names: List[str] = []
    ambiguous_usecases: Set[str] = set()
    uc_conf_threshold = 0.6  # 你可以调整：仅对共识度 < 0.6 的用例做 ASP 过滤

    for name, st in merged.node_stats.items():
        if st.node_type != "usecase":
            continue
        usecase_names.append(name)
        conf = float(st.freq) / float(total)
        if conf < uc_conf_threshold and name not in keep_usecases:
            ambiguous_usecases.add(name)

    # generalize_edges: (child, parent)
    generalize_edges: List[Tuple[str, str]] = []
    for ek in merged.edge_stats.keys():
        if getattr(ek, "kind", "") == "generalize":
            # 约定：src=child dst=parent
            generalize_edges.append((ek.src, ek.dst))

    asp_res = asp_prune_usecases(
        usecases=usecase_names,
        ambiguous_usecases=ambiguous_usecases,
        generalize_edges=generalize_edges,
        keep_usecases=keep_usecases,
        seed=constraints.seed if hasattr(constraints, "seed") and constraints.seed is not None else 7,
    )
    drop_uc_map: Dict[str, List[str]] = asp_res.drop_map  # specific -> [general...]

    # --- Actors ---
    v_actors: List[ValidatedActor] = []
    actor_conflicts: List[Conflict] = []

    for name, st in merged.node_stats.items():
        if st.node_type != "actor":
            continue

        if name in forbid_nodes:
            v_actors.append(
                ValidatedActor(
                    name=name,
                    decision="reject",
                    reasons=["constraints.forbid_actors"],
                    evidence={"freq": st.freq, "sources": st.sources},
                )
            )
            continue

        if name in keep_nodes:
            v_actors.append(
                ValidatedActor(
                    name=name,
                    decision="accept",
                    reasons=["constraints.keep_actors"],
                    evidence={"freq": st.freq, "sources": st.sources},
                )
            )
            continue

        if _mentioned(req_text, name):
            v_actors.append(
                ValidatedActor(
                    name=name,
                    decision="accept",
                    reasons=["需求文本中出现该参与者"],
                    evidence={"freq": st.freq, "sources": st.sources},
                )
            )
        else:
            v_actors.append(
                ValidatedActor(
                    name=name,
                    decision="unknown",
                    reasons=["需求文本未明确出现该参与者，需确认"],
                    evidence={"freq": st.freq, "sources": st.sources},
                )
            )
            actor_conflicts.append(
                Conflict(
                    conflict_type="actor_uncertain",
                    a=name,
                    b="",
                    details={"freq": st.freq / total, "sources": st.sources},
                )
            )

    # --- UseCases ---
    v_usecases: List[ValidatedUseCase] = []
    uc_conflicts: List[Conflict] = []

    # 粒度层次：前缀包含（用于 parent 字段，仅做展示用）
    all_uc = [n for n, st in merged.node_stats.items() if st.node_type == "usecase"]

    for name, st in merged.node_stats.items():
        if st.node_type != "usecase":
            continue

        # [ASP] 若被判定为“被蕴含的具体用例”，直接 reject（除非用户 keep）
        if name in drop_uc_map and name not in keep_usecases:
            v_usecases.append(
                ValidatedUseCase(
                    name=name,
                    decision="reject",
                    reasons=[f"ASP过滤：被更抽象用例蕴含 -> {drop_uc_map[name]}"],
                    evidence={
                        "freq": st.freq,
                        "sources": st.sources,
                        "confidence": st.freq / total,
                        "asp": {"dropped": True, "implied_by": drop_uc_map[name], "used_asp": asp_res.used_asp},
                    },
                    parent=None,
                )
            )
            continue

        if name in forbid_nodes:
            v_usecases.append(
                ValidatedUseCase(
                    name=name,
                    decision="reject",
                    reasons=["constraints.forbid_usecases"],
                    evidence={"freq": st.freq, "sources": st.sources, "confidence": st.freq / total},
                    parent=None,
                )
            )
            continue

        if name in keep_nodes:
            v_usecases.append(
                ValidatedUseCase(
                    name=name,
                    decision="accept",
                    reasons=["constraints.keep_usecases"],
                    evidence={"freq": st.freq, "sources": st.sources, "confidence": st.freq / total},
                    parent=None,
                )
            )
            continue

        parent = None
        n_name = normalize_name(name)
        for p in all_uc:
            if p == name:
                continue
            n_p = normalize_name(p)
            if n_p and n_name.startswith(n_p) and len(n_name) > len(n_p) + 1:
                parent = p
                break

        if _mentioned(req_text, name):
            v_usecases.append(
                ValidatedUseCase(
                    name=name,
                    decision="accept",
                    reasons=["需求文本中出现该用例"],
                    evidence={"freq": st.freq, "sources": st.sources, "confidence": st.freq / total},
                    parent=parent,
                )
            )
        else:
            v_usecases.append(
                ValidatedUseCase(
                    name=name,
                    decision="unknown",
                    reasons=["需求文本未明确出现该用例，需确认"],
                    evidence={"freq": st.freq, "sources": st.sources, "confidence": st.freq / total},
                    parent=parent,
                )
            )
            uc_conflicts.append(
                Conflict(
                    conflict_type="usecase_uncertain",
                    a=name,
                    b="",
                    details={"freq": st.freq / total, "sources": st.sources},
                )
            )

    # --- Relations ---
    v_relations: List[ValidatedRelation] = []
    rel_conflicts: List[Conflict] = []

    must_include = {(normalize_name(x["base"]), normalize_name(x["included"])) for x in constraints.must_include}
    forbid_include = {(normalize_name(x["base"]), normalize_name(x["included"])) for x in constraints.forbid_include}

    must_extend = {(normalize_name(x["base"]), normalize_name(x["extension"])) for x in constraints.extend}
    forbid_extend = {(normalize_name(x["base"]), normalize_name(x["extension"])) for x in constraints.forbid_extend}
    extend_extra = {(normalize_name(x["base"]), normalize_name(x["extension"])): x for x in constraints.extend}

    drop_set = set(drop_uc_map.keys()) - keep_usecases

    # --- [PATCH] extend: 先补全 condition/extension_point，再交给 ASP 检验因果约束 ---
    total = max(int(getattr(merged, "total_candidates", 1)), 1)

    # 仅对“低共识/模糊”的 extend 做 ASP 过滤，避免误伤高共识边
    EXT_CONF_THRESHOLD = 0.6

    extend_items = []
    for ek, es in merged.edge_stats.items():
        if ek.kind != "extend":
            continue
        conf = float(es.freq) / float(total)
        if conf >= EXT_CONF_THRESHOLD:
            continue  # 高共识 extend 不做强过滤

        # meta 里取一条代表性的记录
        cond = ""
        ep = ""
        justification = ""
        for m in (es.meta or []):
            if isinstance(m, dict):
                cond = cond or (m.get("condition") or "")
                ep = ep or (m.get("extension_point") or "")
                justification = justification or (m.get("justification") or "")
        if (not cond) and (not ep) and justification:
            c2, e2 = extract_cond_extpt_from_justification(justification)
            cond = cond or c2
            ep = ep or e2

        key = f"{ek.src}|extend|{ek.dst}"
        extend_items.append(
            {
                "key": key,
                "src": ek.src,
                "dst": ek.dst,
                "condition": (cond or "").strip(),
                "extension_point": (ep or "").strip(),
            }
        )

    asp_ext = asp_check_extend_causality(extend_items)
    bad_extend_keys = set(asp_ext.bad_pairs)


    for ek, es in merged.edge_stats.items():
        evidence = {
            "freq": es.freq,
            "confidence": es.freq / total,
            "sources": es.sources,
            "examples": (es.meta[:2] if es.meta else []),
        }

        # 若关系涉及被 ASP 删除的用例，直接 reject（避免进入最终图）
        if (ek.src in drop_set) or (ek.dst in drop_set):
            v_relations.append(
                ValidatedRelation(
                    src=ek.src,
                    dst=ek.dst,
                    kind=ek.kind,
                    decision="reject",
                    reasons=["ASP过滤：关系涉及被蕴含删除的具体用例"],
                    evidence=evidence,
                )
            )
            continue

        # assoc / generalize 默认 accept
        if ek.kind in ("assoc", "generalize"):
            v_relations.append(
                ValidatedRelation(
                    src=ek.src,
                    dst=ek.dst,
                    kind=ek.kind,
                    decision="accept",
                    reasons=["结构关系默认通过"],
                    evidence=evidence,
                )
            )
            continue

        # include
        if ek.kind == "include":
            key = (normalize_name(ek.src), normalize_name(ek.dst))
            if key in forbid_include:
                v_relations.append(
                    ValidatedRelation(
                        src=ek.src,
                        dst=ek.dst,
                        kind=ek.kind,
                        decision="reject",
                        reasons=["constraints.forbid_include"],
                        evidence=evidence,
                    )
                )
                continue
            if key in must_include:
                v_relations.append(
                    ValidatedRelation(
                        src=ek.src,
                        dst=ek.dst,
                        kind=ek.kind,
                        decision="accept",
                        reasons=["constraints.must_include"],
                        evidence=evidence,
                    )
                )
                continue

            ok, why = _include_heuristic(ek.src, ek.dst)
            if ok:
                v_relations.append(
                    ValidatedRelation(
                        src=ek.src,
                        dst=ek.dst,
                        kind=ek.kind,
                        decision="accept",
                        reasons=[why],
                        evidence=evidence,
                    )
                )
            else:
                v_relations.append(
                    ValidatedRelation(
                        src=ek.src,
                        dst=ek.dst,
                        kind=ek.kind,
                        decision="unknown",
                        reasons=["无法确定 include 必要性，需确认"],
                        evidence=evidence,
                    )
                )
                rel_conflicts.append(
                    Conflict(
                        conflict_type="relation_unknown",
                        a=ek.src,
                        b=ek.dst,
                        details={"freq": es.freq / total, "sources": es.sources},
                    )
                )
            continue

        # extend
        if ek.kind == "extend":
            evidence = {
                "freq": es.freq,
                "confidence": es.freq / total,
                "sources": es.sources,
                "examples": (es.meta[:2] if es.meta else []),
            }

            # 先尝试从 meta 中拿 cond/ep；没有则从 justification 自动抽取
            cond = ""
            ep = ""
            justification = ""
            for m in (es.meta or []):
                if isinstance(m, dict):
                    cond = cond or (m.get("condition") or "")
                    ep = ep or (m.get("extension_point") or "")
                    justification = justification or (m.get("justification") or "")
            if (not cond) and (not ep) and justification:
                c2, e2 = extract_cond_extpt_from_justification(justification)
                cond = cond or c2
                ep = ep or e2

            evidence.update({"condition": cond, "extension_point": ep, "justification": justification})

            key = f"{ek.src}|extend|{ek.dst}"
            if key in bad_extend_keys:
                # 你可以选 reject 或 unknown。若你希望“过滤掉不符合因果逻辑的模糊边”，用 reject 更符合你的目标。
                v_relations.append(
                    ValidatedRelation(
                        src=ek.src,
                        dst=ek.dst,
                        kind=ek.kind,
                        decision="reject",
                        reasons=["ASP因果约束：extend 缺少触发条件/扩展点（condition/extension_point）"],
                        evidence={**evidence, "asp": {"used": asp_ext.used_asp}},
                    )
                )
                rel_conflicts.append(
                    Conflict(
                        conflict_type="extend_missing_condition",
                        a=ek.src,
                        b=ek.dst,
                        details={"freq": es.freq / total, "sources": es.sources},
                    )
                )
                continue

            # 通过 ASP 检查（或规则降级检查）
            v_relations.append(
                ValidatedRelation(
                    src=ek.src,
                    dst=ek.dst,
                    kind=ek.kind,
                    decision="accept",
                    reasons=["extend 具备触发条件/扩展点（或已从 justification 自动抽取）"],
                    evidence={**evidence, "asp": {"used": asp_ext.used_asp}},
                )
            )
            continue

        # 其他未知关系：保守 unknown
        v_relations.append(
            ValidatedRelation(
                src=ek.src,
                dst=ek.dst,
                kind=ek.kind,
                decision="unknown",
                reasons=["未知关系类型，需确认"],
                evidence=evidence,
            )
        )
        rel_conflicts.append(
            Conflict(
                conflict_type="relation_unknown",
                a=ek.src,
                b=ek.dst,
                details={"freq": es.freq / total, "sources": es.sources},
            )
        )

    report = ValidationReport(actors=v_actors, usecases=v_usecases, relations=v_relations)
    conflicts = actor_conflicts + uc_conflicts + rel_conflicts
    return report, conflicts
