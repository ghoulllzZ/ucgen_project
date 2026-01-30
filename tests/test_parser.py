from ucgen.plantuml import parse_plantuml_usecase_diagram


def test_parse_basic():
    puml = """@startuml
actor "User" as U
(Create Shape) as UC1
(Select Shape) as UC2
U --> UC1
UC1 ..> UC2 : <<include>>
@enduml
"""
    ir = parse_plantuml_usecase_diagram(puml)
    assert "User" in ir.actors
    assert "Create Shape" in ir.usecases
    assert "Select Shape" in ir.usecases
    kinds = [r.kind for r in ir.relations]
    assert "assoc" in kinds
    assert "include" in kinds
