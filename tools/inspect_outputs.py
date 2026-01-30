# tools/inspect_outputs.py
from __future__ import annotations

import io
import contextlib
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] File not found: {p}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] JSON parse error in {p} at line {e.lineno}, col {e.colno}: {e.msg}")


def _fmt_sources(sources: Any, maxn: int = 4) -> str:
    if not sources:
        return "-"
    if not isinstance(sources, list):
        return str(sources)
    if len(sources) <= maxn:
        return ",".join(map(str, sources))
    return ",".join(map(str, sources[:maxn])) + f",...(+{len(sources) - maxn})"


def _fmt_node(n: Dict[str, Any]) -> str:
    return f"{n.get('name')} ({n.get('node_type')})"


def _fmt_edge(e: Dict[str, Any]) -> str:
    return f"{e.get('src')} -[{e.get('kind')}]-> {e.get('dst')}"


def _classify_nodes_edges(merged: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    nodes = merged.get("nodes", {}) or {}
    edges = merged.get("edges", {}) or {}

    buckets: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "common": {"nodes": [], "edges": []},
        "conflict": {"nodes": [], "edges": []},
        "unknown": {"nodes": [], "edges": []},
        "other": {"nodes": [], "edges": []},
    }

    for _, nd in nodes.items():
        st = str(nd.get("status", "other")).lower()
        if st not in buckets:
            st = "other"
        buckets[st]["nodes"].append(nd)

    for _, ed in edges.items():
        st = str(ed.get("status", "other")).lower()
        if st not in buckets:
            st = "other"
        buckets[st]["edges"].append(ed)

    # Sort for stable display
    for st in buckets:
        buckets[st]["nodes"].sort(key=lambda x: (x.get("node_type", ""), -int(x.get("freq", 0) or 0), x.get("name", "")))
        buckets[st]["edges"].sort(key=lambda x: (x.get("kind", ""), -int(x.get("freq", 0) or 0), x.get("src", ""), x.get("dst", "")))

    return buckets


def _bucket_validated(validated: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "accept": {"actors": [], "usecases": [], "relations": []},
        "reject": {"actors": [], "usecases": [], "relations": []},
        "unknown": {"actors": [], "usecases": [], "relations": []},
        "other": {"actors": [], "usecases": [], "relations": []},
    }

    for cat in ("actors", "usecases", "relations"):
        for item in (validated.get(cat, []) or []):
            dec = str(item.get("decision", "other")).lower()
            if dec not in out:
                dec = "other"
            out[dec][cat].append(item)

    # Sort for stable display
    out["accept"]["actors"].sort(key=lambda x: x.get("name", ""))
    for dec in out:
        out[dec]["usecases"].sort(key=lambda x: x.get("name", ""))
        out[dec]["relations"].sort(key=lambda x: (x.get("kind", ""), x.get("src", ""), x.get("dst", "")))

    return out


def _split_conflicts(conflicts: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # supports either list[...] or {"conflicts":[...]} in case you change format later
    if isinstance(conflicts, dict) and "conflicts" in conflicts:
        conflicts_list = conflicts.get("conflicts") or []
    else:
        conflicts_list = conflicts or []

    strict: List[Dict[str, Any]] = []
    fuzzy: List[Dict[str, Any]] = []
    for c in conflicts_list:
        ct = str(c.get("conflict_type", "")).lower()
        if ct.endswith("_uncertain") or "missing" in ct or "unknown" in ct:
            fuzzy.append(c)
        else:
            strict.append(c)
    return strict, fuzzy


def _print_section(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect merged_graph.json + conflicts.json + validated.json and print common/different/ambiguous lists."
    )
    ap.add_argument("--outdir", type=str, default=None, help="输出子目录（包含 merged_graph.json/conflicts.json/validated.json）")
    ap.add_argument("--merged", type=str, default=None, help="merged_graph.json 路径")
    ap.add_argument("--conflicts", type=str, default=None, help="conflicts.json 路径")
    ap.add_argument("--validated", type=str, default=None, help="validated.json 路径")
    ap.add_argument("--max-items", type=int, default=80, help="每个清单最多打印多少条")
    ap.add_argument("--save", type=str, default=None, help="将输出写入文件（txt/md），例如 out/diff_report.txt")

    args = ap.parse_args()

    if args.outdir:
        outdir = Path(args.outdir)
        merged_p = outdir / "merged_graph.json"
        conflicts_p = outdir / "conflicts.json"
        validated_p = outdir / "validated.json"
    else:
        merged_p = Path(args.merged) if args.merged else None
        conflicts_p = Path(args.conflicts) if args.conflicts else None
        validated_p = Path(args.validated) if args.validated else None

    if not (merged_p and conflicts_p and validated_p):
        raise SystemExit(
            "[ERROR] Please provide either --outdir <dir> OR all of --merged/--conflicts/--validated."
        )

    merged = _read_json(merged_p)
    conflicts = _read_json(conflicts_p)
    validated = _read_json(validated_p)

    # 如果指定 --save，则把所有 print 输出写入文件；否则正常打印到终端
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            # ===== 把下面原来的所有打印逻辑放在这里（整体缩进）=====
            stats = (merged.get("stats") or {}) if isinstance(merged, dict) else {}
            print("=== 输出诊断汇总 ===")
            print(
                f"candidates={stats.get('total_candidates')} "
                f"nodes={stats.get('nodes')} "
                f"edges={stats.get('edges')} "
                f"conflicts={stats.get('conflicts')}"
            )

            buckets = _classify_nodes_edges(merged)
            vb = _bucket_validated(validated)
            strict_conf, fuzzy_conf = _split_conflicts(conflicts)

            # 1) common
            _print_section("1) 相同部分（共识/已通过验证）")

            print("[Stage2 共识节点]")
            for n in buckets["common"]["nodes"][: args.max_items]:
                print(f" - {_fmt_node(n)} | freq={n.get('freq')} | sources={_fmt_sources(n.get('sources'))}")

            print("\n[Stage2 共识关系]")
            for e in buckets["common"]["edges"][: args.max_items]:
                print(f" - {_fmt_edge(e)} | freq={e.get('freq')} | sources={_fmt_sources(e.get('sources'))}")

            print("\n[Stage3 验证通过（accept）]")
            print(
                f" - actors={len(vb['accept']['actors'])} "
                f"usecases={len(vb['accept']['usecases'])} "
                f"relations={len(vb['accept']['relations'])}"
            )
            for r in vb["accept"]["relations"][: args.max_items]:
                ev = r.get("evidence", {}) or {}
                print(
                    f" - {r.get('src')} -[{r.get('kind')}]-> {r.get('dst')} "
                    f"| conf={ev.get('confidence')} freq={ev.get('freq')} sources={_fmt_sources(ev.get('sources'))}"
                )

            # 2) different
            _print_section("2) 不同部分（候选之间的明确冲突）")
            if not strict_conf:
                print("（无：当前冲突主要表现为“不确定/缺条件”，见第 3 部分）")
            else:
                for c in strict_conf[: args.max_items]:
                    print(f" - [{c.get('conflict_type')}] {c.get('a')} vs {c.get('b')} | details={c.get('details')}")

            # 3) ambiguous
            _print_section("3) 模糊部分（待确认/置信度不足/缺触发条件）")

            print("[Stage2 待确认节点（status=unknown）]")
            for n in buckets["unknown"]["nodes"][: args.max_items]:
                print(f" - {_fmt_node(n)} | freq={n.get('freq')} | sources={_fmt_sources(n.get('sources'))}")

            print("\n[Stage2 待确认关系（status=unknown）]")
            for e in buckets["unknown"]["edges"][: args.max_items]:
                meta = e.get("meta") or []
                hint = ""
                if meta and isinstance(meta, list):
                    ex = meta[0] or {}
                    j = ex.get("justification", "") or ""
                    cond = ex.get("condition", "") or ""
                    ep = ex.get("extension_point", "") or ""
                    if j or cond or ep:
                        hint = f" | hint: has_justif={bool(j)} has_cond={bool(cond)} has_extpt={bool(ep)}"
                print(f" - {_fmt_edge(e)} | freq={e.get('freq')} | sources={_fmt_sources(e.get('sources'))}{hint}")

            print("\n[Stage3 判定为 unknown 的关系]")
            print(
                f" - usecases(unknown)={len(vb['unknown']['usecases'])} relations(unknown)={len(vb['unknown']['relations'])}")
            for r in vb["unknown"]["relations"][: args.max_items]:
                ev = r.get("evidence", {}) or {}
                reasons = r.get("reasons") or []
                reasons_str = "; ".join([str(x) for x in reasons[:2]])
                print(
                    f" - {r.get('src')} -[{r.get('kind')}]-> {r.get('dst')} "
                    f"| reasons={reasons_str} | conf={ev.get('confidence')} freq={ev.get('freq')}"
                )

            print("\n[conflicts.json 中的“不确定/缺条件”】【列表】]")
            for c in fuzzy_conf[: args.max_items]:
                det = c.get("details") or {}
                extra = []
                if "confidence" in det:
                    extra.append(f"conf={det.get('confidence')}")
                if "msg" in det and det.get("msg"):
                    extra.append(f"msg={det.get('msg')}")
                if "sources" in det and det.get("sources"):
                    extra.append(f"sources={_fmt_sources(det.get('sources'))}")
                extra_str = (" | " + " ".join(extra)) if extra else ""
                print(f" - [{c.get('conflict_type')}] {c.get('a')} -> {c.get('b')}{extra_str}")
        return




if __name__ == "__main__":
    main()
