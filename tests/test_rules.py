from ucgen.merge import merge_candidates
from ucgen.plantuml import parse_plantuml_usecase_diagram
from ucgen.reasoning import PythonRuleEngine
from ucgen.types import Constraints


def test_rule_engine_include_heuristic_keeps_select_dependency():
    c1 = """@startuml
actor "用户" as U
(删除图形) as A
(选择图形) as B
U --> A
U --> B
A ..> B : <<include>>
@enduml
"""
    ir1 = parse_plantuml_usecase_diagram(c1)
    merged = merge_candidates({"c1": ir1}, consensus_threshold=0.6)
    engine = PythonRuleEngine()
    decision = engine.decide(merged, "删除图形前请先选择图形。", Constraints())
    kept = {(k.src, k.kind, k.dst) for k in decision.kept_edges}
    assert ("删除图形", "include", "选择图形") in kept
