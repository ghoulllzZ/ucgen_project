from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple

import networkx as nx

from .types import Conflict, DiagramIR, EdgeKey, EdgeStats, NodeStats, RelationKind

try:
    from rapidfuzz.fuzz import ratio as _rf_ratio  # type: ignore

    def _ratio(a: str, b: str) -> int:
        return int(_rf_ratio(a, b))
except Exception:
    def _ratio(a: str, b: str) -> int:
        return int(SequenceMatcher(None, a, b).ratio() * 100)


_SYN = {
    "形状": "图形",
    "大小": "尺寸",
    "改变尺寸": "调整尺寸",
    "改变图形尺寸": "调整图形尺寸",
}


def normalize_name(name: str) -> str:
    """NFKC + 去空白 + 同义替换 + 标点弱化。"""
    s = unicodedata.normalize("NFKC", str(name))
    s = "".join(s.split())
    for k, v in _SYN.items():
        s = s.replace(k, v)
    s = re.sub(r"[，,。．\.！!？?\-_/\\]+", "", s)
    return s


def align_name(name: str, canon_pool: List[str], threshold: int = 90) -> str:
    """将 name 对齐到 canon_pool 中最相近者，否则自己成为新 canonical。"""
    n = normalize_name(name)
    if not canon_pool:
        canon_pool.append(n)
        return n
    best = None
    best_score = -1
    for c in canon_pool:
        sc = _ratio(n, c)
        if sc > best_score:
            best_score = sc
            best = c
    if best is not None and best_score >= threshold:
        return best
    canon_pool.append(n)
    return n


@dataclass
class MergedModel:
    total_candidates: int
    graph: nx.MultiDiGraph
    node_stats: Dict[str, NodeStats]
    edge_stats: Dict[EdgeKey, EdgeStats]
    conflicts: List[Conflict]


def merge_candidates(
    candidates: Dict[str, DiagramIR],
    consensus_threshold: float = 0.6,
    fuzzy_threshold: int = 90,
) -> MergedModel:
    total = len(candidates)
    g = nx.MultiDiGraph()

    node_stats: Dict[str, NodeStats] = {}
    edge_stats: Dict[EdgeKey, EdgeStats] = {}

    # canonical 池（fuzzy 对齐）
    canon_pool: List[str] = []

    # (src,dst) -> kinds set
    edge_kinds_by_pair: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    edge_freq_any_by_pair: Dict[Tuple[str, str], int] = defaultdict(int)
    edge_sources_any_by_pair: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    def _add_source(lst: List[str], cid: str) -> None:
        if cid not in lst:
            lst.append(cid)

    # 先建 node
    for cid, ir in candidates.items():
        for a in ir.actors:
            n = align_name(a, canon_pool, threshold=fuzzy_threshold)
            st = node_stats.get(n)
            if not st:
                st = NodeStats(name=n, node_type="actor", freq=0, sources=[], status="unknown")
                node_stats[n] = st
            st.freq += 1
            _add_source(st.sources, cid)
            g.add_node(n, node_type="actor")

        for u in ir.usecases:
            n = align_name(u, canon_pool, threshold=fuzzy_threshold)
            st = node_stats.get(n)
            if not st:
                st = NodeStats(name=n, node_type="usecase", freq=0, sources=[], status="unknown")
                node_stats[n] = st
            st.freq += 1
            _add_source(st.sources, cid)
            g.add_node(n, node_type="usecase")

    # 再建 edge
    for cid, ir in candidates.items():
        for r in ir.relations:
            src = align_name(r.src, canon_pool, threshold=fuzzy_threshold)
            dst = align_name(r.dst, canon_pool, threshold=fuzzy_threshold)
            kind: RelationKind = r.kind

            key = EdgeKey(src=src, kind=kind, dst=dst)
            es = edge_stats.get(key)
            if not es:
                es = EdgeStats(src=src, kind=kind, dst=dst, freq=0, sources=[], status="unknown", meta=[])
                edge_stats[key] = es
            es.freq += 1
            _add_source(es.sources, cid)

            es.meta.append({
                "candidate_id": cid,
                "line_no": r.line_no,
                "raw": r.raw,
                "label": r.label,
                "justification": r.justification,
                "condition": r.condition,
                "extension_point": r.extension_point,
                "provenance": r.provenance,
            })

            g.add_edge(src, dst, key=kind, kind=kind, candidate_id=cid, line_no=r.line_no)

            if kind in ("include", "extend"):
                edge_kinds_by_pair[(src, dst)].add(kind)
                edge_freq_any_by_pair[(src, dst)] += 1
                edge_sources_any_by_pair[(src, dst)].add(cid)

    # 标注 common/unknown
    # 严格定义：相同(common) 必须是“所有候选图都出现”的元素（交集）
    #   - common: freq == total（所有候选都出现，严格交集）
    # - different: freq == 1（仅一个候选出现）
    # - fuzzy: 1 < freq < total（部分候选出现）
    # for _, st in node_stats.items():
        # conf = st.freq / max(total, 1)
        # st.status = "common" if conf >= consensus_threshold else "unknown"
        # st.status = "common" if st.freq == total else "unknown"
        # st.sources.sort()
    for _, st in node_stats.items():
        if st.freq == total:
            st.status = "common"
        elif st.freq == 1:
            st.status = "different"
        else:
            st.status = "fuzzy"
        st.sources.sort()
    conflicts: List[Conflict] = []

    # include vs extend 冲突
    for (a, b), kinds in edge_kinds_by_pair.items():
        if len(kinds) >= 2:
            support = {}
            for k in kinds:
                ek = EdgeKey(a, k, b)
                support[k] = edge_stats.get(ek).freq if edge_stats.get(ek) else 0
            conflicts.append(
                Conflict(
                    conflict_type="edge_kind_mismatch",
                    a=a,
                    b=b,
                    details={"kinds": sorted(list(kinds)), "support": support},
                )
            )

    # 出现频次不够（待确认）
    # 注意：现在“模糊(fuzzy)”才算待确认：1 < freq_any < total
    for (a, b), freq_any in edge_freq_any_by_pair.items():
        conf = freq_any / max(total, 1)
        # if 0 < conf < consensus_threshold:
        # 严格定义：只要不是所有候选都出现，就视为“存在性待确认”
        if 1 < freq_any < total:
            conflicts.append(
                Conflict(
                    conflict_type="edge_presence_uncertain",
                    a=a,
                    b=b,
                    #details={"confidence": conf, "sources": sorted(list(edge_sources_any_by_pair[(a, b)]))},
                    details={
                            "confidence": conf,
                            "bucket": "fuzzy",
                            "freq": freq_any,
                            "total": total,
                            "sources": sorted(list(edge_sources_any_by_pair[(a, b)])),
                        },
                )
            )

    # extend 缺少条件/扩展点（阶段2就能发现）
    for ek, es in edge_stats.items():
        conf = es.freq / max(total, 1)
        if ek.kind in ("include", "extend") and (ek.src, ek.dst) in edge_kinds_by_pair and len(edge_kinds_by_pair[(ek.src, ek.dst)]) >= 2:
            es.status = "conflict"
        else:
            # es.status = "common" if conf >= consensus_threshold else "unknown"
            # es.status = "common" if es.freq == total else "unknown"
            if es.freq == total:
                es.status = "common"
            elif es.freq == 1:
                es.status = "different"
            else:
                es.status = "fuzzy"

        if ek.kind == "extend":
            has_cond = False
            for m in es.meta:
                if str(m.get("condition", "")).strip() or str(m.get("extension_point", "")).strip():
                    has_cond = True
                    break
            if not has_cond:
                conflicts.append(
                    Conflict(
                        conflict_type="extend_missing_condition",
                        a=ek.src,   # extension
                        b=ek.dst,   # base
                        details={"msg": "extend 缺少触发条件/扩展点"},
                    )
                )

        es.sources.sort()

    return MergedModel(total_candidates=total, graph=g, node_stats=node_stats, edge_stats=edge_stats, conflicts=conflicts)


def jaccard_similarity_edges(a: Set[Tuple[str, str, str]], b: Set[Tuple[str, str, str]]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(union, 1)
