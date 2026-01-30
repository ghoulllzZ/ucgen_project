from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field

RelationKind = Literal["assoc", "include", "extend", "generalize"]


class PlantUMLParseError(Exception):
    def __init__(self, message: str, line_no: int, line: str):
        super().__init__(f"{message} (line {line_no}): {line}")
        self.message = message
        self.line_no = line_no
        self.line = line


@dataclass(frozen=True)
class Relation:
    """
    解析后的关系结构化字段（阶段2/3会用到）：
    - justification: 因果解释（来自紧邻关系行前的注释）
    - condition/extension_point: extend 的触发条件/扩展点（可从注释/label抽取）
    """
    src: str
    kind: RelationKind
    dst: str
    label: str = ""
    raw: str = ""
    line_no: int = 0
    justification: str = ""
    condition: str = ""
    extension_point: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagramIR:
    actors: Set[str]
    usecases: Set[str]
    relations: List[Relation]
    alias_to_name: Dict[str, str]


@dataclass(frozen=True)
class EdgeKey:
    src: str
    kind: RelationKind
    dst: str


class NodeStats(BaseModel):
    name: str
    node_type: Literal["actor", "usecase"]
    freq: int = 0
    # 注意：为保证 JSON 可序列化，不使用 set
    sources: List[str] = Field(default_factory=list)
    status: Literal["common", "unknown"] = "unknown"


class EdgeStats(BaseModel):
    src: str
    kind: RelationKind
    dst: str
    freq: int = 0
    sources: List[str] = Field(default_factory=list)
    status: Literal["common", "unknown", "conflict"] = "unknown"
    meta: List[Dict[str, Any]] = Field(default_factory=list)


class Conflict(BaseModel):
    # 扩展冲突类型：支持 actor/usecase/关系 unknown 的交互澄清
    conflict_type: Literal[
        "edge_kind_mismatch",
        "edge_presence_uncertain",
        "extend_missing_condition",
        "actor_uncertain",
        "usecase_uncertain",
        "relation_unknown",
    ]
    a: str
    b: str
    details: Dict[str, Any] = Field(default_factory=dict)


class Constraints(BaseModel):
    # 关系约束（原有）
    must_include: List[Dict[str, str]] = Field(default_factory=list)   # {"base": "...", "included": "..."}
    forbid_include: List[Dict[str, str]] = Field(default_factory=list)
    extend: List[Dict[str, str]] = Field(default_factory=list)         # {"base":"...","extension":"...","condition":"...","extension_point":"..."}
    forbid_extend: List[Dict[str, str]] = Field(default_factory=list)

    # 新增：Actor / UseCase 保留/剔除约束（阶段3 + 交互回灌）
    keep_actors: List[str] = Field(default_factory=list)
    forbid_actors: List[str] = Field(default_factory=list)
    keep_usecases: List[str] = Field(default_factory=list)
    forbid_usecases: List[str] = Field(default_factory=list)

    def normalize_inplace(self) -> None:
        def _norm(s: str) -> str:
            return "".join(str(s).split())

        self.must_include = [{"base": _norm(x["base"]), "included": _norm(x["included"])} for x in self.must_include]
        self.forbid_include = [{"base": _norm(x["base"]), "included": _norm(x["included"])} for x in self.forbid_include]

        ext_out: List[Dict[str, str]] = []
        for x in self.extend:
            ext_out.append({
                "base": _norm(x["base"]),
                "extension": _norm(x["extension"]),
                "condition": str(x.get("condition", "")).strip(),
                "extension_point": str(x.get("extension_point", "")).strip(),
            })
        self.extend = ext_out

        self.forbid_extend = [{"base": _norm(x["base"]), "extension": _norm(x["extension"])} for x in self.forbid_extend]

        self.keep_actors = [_norm(x) for x in self.keep_actors]
        self.forbid_actors = [_norm(x) for x in self.forbid_actors]
        self.keep_usecases = [_norm(x) for x in self.keep_usecases]
        self.forbid_usecases = [_norm(x) for x in self.forbid_usecases]


class Candidate(BaseModel):
    candidate_id: str
    model: str
    temperature: float
    seed: int
    plantuml: str
    parse_error: Optional[str] = None


class IterationReport(BaseModel):
    iter_index: int
    candidates: List[Candidate]
    merged_stats: Dict[str, Any]
    conflicts: List[Conflict]
    questions: List[Dict[str, Any]]
    answers: Dict[str, Any]
    rule_report: Dict[str, Any]
    converged: bool


class RunReport(BaseModel):
    args: Dict[str, Any]
    constraints: Dict[str, Any]
    iterations: List[IterationReport]
    final: Dict[str, Any]


@dataclass
class RunArgs:
    req_path: Path
    out_dir: Path
    max_iters: int
    backend: str
    temperature: float
    n_samples: int
    models: List[str]
    seed: int
    interactive: bool
    jaccard_threshold: float
    render: str
    config_path: Optional[Path]


# 阶段3输出：validated.json
Decision = Literal["accept", "reject", "unknown"]


class ValidatedActor(BaseModel):
    name: str
    decision: Decision
    reasons: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)


class ValidatedUseCase(BaseModel):
    name: str
    decision: Decision
    reasons: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    # 可选：粒度/层次建议（generalization）
    parent: Optional[str] = None


class ValidatedRelation(BaseModel):
    src: str
    kind: RelationKind
    dst: str
    decision: Decision
    reasons: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    actors: List[ValidatedActor] = Field(default_factory=list)
    usecases: List[ValidatedUseCase] = Field(default_factory=list)
    relations: List[ValidatedRelation] = Field(default_factory=list)
