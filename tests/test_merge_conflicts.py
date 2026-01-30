from ucgen.merge import merge_candidates
from ucgen.plantuml import parse_plantuml_usecase_diagram


def test_conflict_kind_mismatch():
    c1 = """@startuml
actor "用户" as U
(移动图形) as A
(选择图形) as B
U --> A
U --> B
A ..> B : <<include>>
@enduml
"""
    c2 = """@startuml
actor "用户" as U
(移动图形) as A
(选择图形) as B
U --> A
U --> B
A ..> B : <<extend>>
@enduml
"""
    ir1 = parse_plantuml_usecase_diagram(c1)
    ir2 = parse_plantuml_usecase_diagram(c2)
    merged = merge_candidates({"c1": ir1, "c2": ir2}, consensus_threshold=0.6)
    assert any(c.conflict_type == "edge_kind_mismatch" for c in merged.conflicts)
