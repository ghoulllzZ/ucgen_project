from __future__ import annotations
from .llm_adapters import build_llm_from_config


import json
import logging
import random
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

from .config import load_config
from .interaction import apply_answers_to_constraints, collect_answers, generate_questions
from .logging_utils import time_block
from .merge import MergedModel, jaccard_similarity_edges, merge_candidates
from .plantuml import generate_plantuml, parse_plantuml_usecase_diagram
from .prompt_builder import build_prompt
from .types import Candidate, Constraints, DiagramIR, EdgeKey, IterationReport, RunArgs, RunReport, Relation
from .validator import validate_all

logger = logging.getLogger("ucgen")


class BaseLLM:
    name: str

    def generate(self, prompt: str, temperature: float, seed: int) -> str:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """默认可用 MockLLM：根据随机种子生成不同候选（确保差异性）。"""

    def __init__(self, name: str = "mock"):
        self.name = name

    def _extract_constraints(self, prompt: str) -> Constraints:
        try:
            jstart = prompt.rfind("{")
            jend = prompt.rfind("}")
            payload = json.loads(prompt[jstart : jend + 1])
            c = Constraints(**payload.get("constraints", {}))
            c.normalize_inplace()
            return c
        except Exception:
            return Constraints()

    def generate(self, prompt: str, temperature: float, seed: int) -> str:
        rnd = random.Random(seed)
        constraints = self._extract_constraints(prompt)

        user = "用户"
        ucs = ["创建图形", "选择图形", "移动图形", "删除图形", "调整图形尺寸"]
        if rnd.random() < 0.6:
            ucs.append("等比例缩放")

        include_edges: List[Tuple[str, str, str]] = []
        extend_edges: List[Tuple[str, str, str, str, str]] = []

        must_i = {(x["base"], x["included"]) for x in constraints.must_include}
        forb_i = {(x["base"], x["included"]) for x in constraints.forbid_include}
        must_e = {(x["extension"], x["base"]): x.get("condition", "") for x in constraints.extend}
        forb_e = {(x["extension"], x["base"]) for x in constraints.forbid_extend}

        default_pairs = [
            ("移动图形", "选择图形"),
            ("删除图形", "选择图形"),
            ("调整图形尺寸", "选择图形"),
        ]

        for base, inc in default_pairs:
            if (base, inc) in forb_i:
                continue
            if (base, inc) in must_i:
                include_edges.append((base, inc, "用户约束：必须先选择目标图形"))
                continue

            p = rnd.random()
            if p < 0.55:
                include_edges.append((base, inc, "执行前需要锁定/选中目标"))
            elif p < 0.75:
                extend_edges.append(
                    (base, inc, "仅在特定条件下才选择（争议示例）", "当拖拽前未选中任何图形时", "拖拽开始前")
                )
            else:
                pass

        if "等比例缩放" in ucs:
            ext = "等比例缩放"
            base = "调整图形尺寸"
            nk = (ext, base)
            if nk not in forb_e:
                if nk in must_e:
                    cond = must_e[nk].strip() or "按住Shift时"
                    extend_edges.append((ext, base, "在特定条件下触发等比例缩放", cond, "拖拽控制点时"))
                else:
                    if rnd.random() < 0.65:
                        extend_edges.append((ext, base, "按住Shift时触发等比例缩放", "按住Shift时", "拖拽控制点时"))

        lines: List[str] = []
        lines.append("@startuml")
        lines.append("left to right direction")
        lines.append("skinparam shadowing false")
        lines.append(f'actor "{user}" as U')
        for i, uc in enumerate(ucs, start=1):
            lines.append(f"({uc}) as UC{i}")

        for i in range(1, len(ucs) + 1):
            lines.append(f"U --> UC{i}")

        def _uc_alias(name: str) -> str:
            idx = ucs.index(name) + 1
            return f"UC{idx}"

        for base, inc, reason in include_edges:
            lines.append(f"' 因果说明：{base} 过程中必须包含 {inc}（{reason}）")
            lines.append(f"{_uc_alias(base)} ..> {_uc_alias(inc)} : <<include>>")

        for ext, base, _, cond, extpt in extend_edges:
            lines.append(f"' 因果说明：{ext} 是对 {base} 的条件扩展；触发条件：{cond}；扩展点：{extpt}")
            lines.append(f"{_uc_alias(ext)} ..> {_uc_alias(base)} : <<extend>>  condition={cond}")

        lines.append("@enduml")
        return "\n".join(lines) + "\n"


class RealLLMSkeleton(BaseLLM):
    """真实 LLM 适配器骨架：未配置 Key 时抛错并由主流程 fallback 到 MockLLM。"""

    def __init__(self, name: str, env_key: str):
        self.name = name
        self.env_key = env_key

    def generate(self, prompt: str, temperature: float, seed: int) -> str:
        import os

        if not os.environ.get(self.env_key, ""):
            raise RuntimeError(f"{self.name} not configured: missing env {self.env_key}")
        raise RuntimeError(f"{self.name} adapter skeleton: please implement API call or use backend=mock")


def get_llms(backend: str, models: list[str], cfg: dict) -> list:
    """
    backend:
      - mock/openai/deepseek/qwen/gemini/doubao/grok/multi
    multi:
      - 默认按 models 顺序构造多个 LLM（如 ["openai","deepseek","qwen","gemini","doubao","grok"]）
    真实 LLM 缺 key/缺依赖时：由上层捕获异常并 fallback 到 MockLLM。
    """
    backend = (backend or "mock").lower()

    if backend == "mock":
        return [MockLLM("mock")]

    if backend in ("openai", "deepseek", "qwen", "gemini", "doubao", "grok"):
        return [build_llm_from_config(backend, cfg)]

    if backend == "multi":
        names = models or ["openai", "deepseek", "qwen", "doubao", "grok"]
        llms = []
        for n in names:
            llms.append(build_llm_from_config(n.lower(), cfg))
        return llms

    # 未知后端默认 mock
    return [MockLLM("mock")]




def _json_default(o: Any) -> Any:
    if isinstance(o, set):
        return sorted(list(o))
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _build_final_ir(merged: MergedModel, kept_edges: Set[EdgeKey], constraints: Constraints) -> DiagramIR:
    """
    节点保留策略：
    - common 节点
    - 或与 kept_edges 有 incident
    - 或在 constraints.keep_* 中
    同时剔除 constraints.forbid_*
    """
    constraints.normalize_inplace()
    keep_nodes = set(constraints.keep_actors + constraints.keep_usecases)
    forbid_nodes = set(constraints.forbid_actors + constraints.forbid_usecases)

    incident: Set[str] = set()
    for ek in kept_edges:
        incident.add(ek.src)
        incident.add(ek.dst)

    actors: Set[str] = set()
    usecases: Set[str] = set()

    for n, st in merged.node_stats.items():
        if n in forbid_nodes:
            continue
        if st.status == "common" or n in incident or n in keep_nodes:
            if st.node_type == "actor":
                actors.add(n)
            else:
                usecases.add(n)

    rels: List[Relation] = []
    for ek in kept_edges:
        if ek.src in forbid_nodes or ek.dst in forbid_nodes:
            continue
        if (ek.src in actors or ek.src in usecases) and (ek.dst in actors or ek.dst in usecases):
            rels.append(Relation(src=ek.src, kind=ek.kind, dst=ek.dst))

    return DiagramIR(actors=actors, usecases=usecases, relations=rels, alias_to_name={})


def run_pipeline(args: RunArgs) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    requirement_text = args.req_path.read_text(encoding="utf-8")

    cfg = load_config(args.config_path)

    constraints = Constraints()
    constraints_path = args.out_dir / "constraints.json"
    if constraints_path.exists():
        try:
            constraints = Constraints(**json.loads(constraints_path.read_text(encoding="utf-8")))
        except Exception:
            constraints = Constraints()

    llms = get_llms(args.backend, args.models, cfg)


    iterations: List[IterationReport] = []
    prev_edge_set: Optional[Set[Tuple[str, str, str]]] = None

    for it in range(1, args.max_iters + 1):
        iter_dir = args.out_dir / f"iter_{it}"
        cand_dir = iter_dir / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)

        # 每轮把 ASP program 写到当前迭代目录（由 asp_reasoner.py 读取该环境变量写盘）
        os.environ["UCGEN_ASP_DUMP_DIR"] = str(iter_dir / "asp")

        logger.info("=== Iteration %d/%d ===", it, args.max_iters)

        conflict_summary = iterations[-1].conflicts if iterations else []
        prompt = build_prompt(requirement_text, constraints, conflict_summary)

        candidates: List[Candidate] = []
        parsed_map: Dict[str, DiagramIR] = {}

        with time_block(logger, "candidate_generation"):
            for llm in llms:
                for j in range(args.n_samples):
                    seed = args.seed + it * 1000 + j * 17 + (hash(llm.name) % 97)
                    try:
                        puml = llm.generate(prompt, temperature=args.temperature, seed=seed)
                    except Exception as e:
                        logger.warning("LLM %s failed (%s); fallback to MockLLM", llm.name, e)
                        puml = MockLLM("fallback-mock").generate(prompt, temperature=args.temperature, seed=seed)

                    c = Candidate(
                        candidate_id=f"it{it}_{llm.name}_{j+1}",
                        model=llm.name,
                        temperature=args.temperature,
                        seed=seed,
                        plantuml=puml,
                    )
                    (cand_dir / f"{c.candidate_id}.puml").write_text(puml, encoding="utf-8")
                    candidates.append(c)

                    try:
                        ir = parse_plantuml_usecase_diagram(
                            puml, provenance={"candidate_id": c.candidate_id, "model": c.model}
                        )
                        parsed_map[c.candidate_id] = ir
                    except Exception as pe:
                        c.parse_error = str(pe)

        # ---------------- Stage2: merge + conflicts ----------------
        merged = merge_candidates(parsed_map, consensus_threshold=0.6, fuzzy_threshold=90)
        merged_stats = {
            "total_candidates": merged.total_candidates,
            "nodes": len(merged.node_stats),
            "edges": len(merged.edge_stats),
            "conflicts": len(merged.conflicts),
        }

        _write_json(
            iter_dir / "merged_graph.json",
            {
                "stats": merged_stats,
                "nodes": {k: v.model_dump(mode="json") for k, v in merged.node_stats.items()},
                "edges": {f"{k.src}|{k.kind}|{k.dst}": v.model_dump(mode="json") for k, v in merged.edge_stats.items()},
            },
        )
        _write_json(iter_dir / "conflicts.json", [c.model_dump(mode="json") for c in merged.conflicts])

        # ---------------- Stage3: validate ----------------
        v_report, v_conflicts = validate_all(requirement_text, merged, constraints)
        _write_json(iter_dir / "validated.json", v_report.model_dump(mode="json"))

        validated_map: Dict[Tuple[str, str, str], str] = {}
        for vr in v_report.relations:
            validated_map[(vr.src, vr.kind, vr.dst)] = vr.decision

        # 决策：accept + must_* 保留；reject + forbid_* 剔除；unknown 交互再定
        constraints.normalize_inplace()
        must_include = {(x["base"], x["included"]) for x in constraints.must_include}
        forbid_include = {(x["base"], x["included"]) for x in constraints.forbid_include}
        must_extend = {(x["extension"], x["base"]) for x in constraints.extend}
        forbid_extend = {(x["extension"], x["base"]) for x in constraints.forbid_extend}

        kept_edges: Set[EdgeKey] = set()
        for ek in merged.edge_stats.keys():
            d = validated_map.get((ek.src, ek.kind, ek.dst), "unknown")

            if ek.kind == "include":
                if (ek.src, ek.dst) in forbid_include:
                    continue
                if (ek.src, ek.dst) in must_include:
                    kept_edges.add(ek)
                    continue
            if ek.kind == "extend":
                if (ek.src, ek.dst) in forbid_extend:
                    continue
                if (ek.src, ek.dst) in must_extend:
                    kept_edges.add(ek)
                    continue

            if d == "accept":
                kept_edges.add(ek)

        # 交互问题来自：阶段2冲突 + 阶段3 unknown 点
        unresolved = merged.conflicts + v_conflicts
        questions = generate_questions(unresolved)
        _write_json(iter_dir / "questions.json", questions)

        answers = collect_answers(questions, interactive=args.interactive)
        _write_json(iter_dir / "answers.json", answers)

        constraints = apply_answers_to_constraints(constraints, questions, answers)
        _write_json(constraints_path, constraints.model_dump(mode="json"))

        # 收敛判据：kept_edges 的 jaccard
        resolved_edge_set = {(k.src, k.kind, k.dst) for k in kept_edges}
        sim = jaccard_similarity_edges(prev_edge_set or set(), resolved_edge_set) if prev_edge_set is not None else 0.0
        converged = False
        if prev_edge_set is not None and sim >= args.jaccard_threshold:
            converged = True
        if not unresolved:
            converged = True
        prev_edge_set = resolved_edge_set

        logger.info(
            "Iteration stats: candidates=%d nodes=%d edges=%d conflicts(stage2)=%d conflicts(stage3)=%d questions=%d jaccard=%.3f",
            len(candidates),
            merged_stats["nodes"],
            merged_stats["edges"],
            len(merged.conflicts),
            len(v_conflicts),
            len(questions),
            sim,
        )

        iterations.append(
            IterationReport(
                iter_index=it,
                candidates=candidates,
                merged_stats=merged_stats,
                conflicts=unresolved,
                questions=questions,
                answers=answers,
                rule_report={
                    "kept_edges": sorted([f"{k.src}|{k.kind}|{k.dst}" for k in kept_edges]),
                    "notes": {"decision": "kept_edges from validated + constraints"},
                },
                converged=converged,
            )
        )

        if converged:
            logger.info("Converged at iteration %d.", it)
            break

    # ---------------- final output ----------------
    # 以最后一轮的 kept_edges 构造 final_ir 并输出 PlantUML
    last_kept = set()
    for s in iterations[-1].rule_report.get("kept_edges", []):
        src, kind, dst = s.split("|", 2)
        last_kept.add(EdgeKey(src=src, kind=kind, dst=dst))  # type: ignore

    # 用最后一轮 candidates 再 merge 一次确保 node_stats 可用
    last_iter = iterations[-1].iter_index
    last_dir = args.out_dir / f"iter_{last_iter}"
    parsed_map_final: Dict[str, DiagramIR] = {}
    for p in (last_dir / "candidates").glob("*.puml"):
        try:
            parsed_map_final[p.stem] = parse_plantuml_usecase_diagram(p.read_text(encoding="utf-8"), provenance={"candidate_id": p.stem})
        except Exception:
            continue
    merged_final = merge_candidates(parsed_map_final, consensus_threshold=0.6, fuzzy_threshold=90)

    final_ir = _build_final_ir(merged_final, last_kept, constraints)
    final_puml = generate_plantuml(final_ir, title="Final Use Case Diagram")
    final_path = args.out_dir / "final_usecase.puml"
    final_path.write_text(final_puml, encoding="utf-8")

    report = RunReport(
        args={
            "req": str(args.req_path),
            "out": str(args.out_dir),
            "max_iters": args.max_iters,
            "backend": args.backend,
            "temperature": args.temperature,
            "n_samples": args.n_samples,
            "models": args.models,
            "seed": args.seed,
            "interactive": args.interactive,
            "jaccard_threshold": args.jaccard_threshold,
            "render": args.render,
        },
        constraints=constraints.model_dump(mode="json"),
        iterations=iterations,
        final={
            "final_usecase_puml": str(final_path),
            "nodes": sorted(list(final_ir.actors | final_ir.usecases)),
            "edges": sorted([f"{r.src}|{r.kind}|{r.dst}" for r in final_ir.relations]),
        },
    )
    (args.out_dir / "report.json").write_text(report.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")

    # 可选渲染（无 java/plantuml 时会自动跳过）
    from .render import render_plantuml
    render_plantuml(final_path, mode=args.render)

    logger.info("Done. Final PlantUML: %s", final_path)
