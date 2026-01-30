from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from .types import DiagramIR, PlantUMLParseError, Relation

_RE_START = re.compile(r"^\s*@startuml\b", re.IGNORECASE)
_RE_END = re.compile(r"^\s*@enduml\b", re.IGNORECASE)

_RE_ACTOR = re.compile(r'^\s*actor\s+(".*?"|[^"\s]+)(?:\s+as\s+([A-Za-z_]\w*))?\s*$', re.IGNORECASE)
_RE_UC_PAREN = re.compile(r'^\s*\((.+?)\)\s*(?:as\s+([A-Za-z_]\w*))?\s*$', re.IGNORECASE)
_RE_UC_KW = re.compile(r'^\s*usecase\s+(".*?"|[^"\s]+)(?:\s+as\s+([A-Za-z_]\w*))?\s*$', re.IGNORECASE)

_RE_REL = re.compile(r'^\s*(.+?)\s+([.<-]{1,3}[.-]*[>-]{1,3}|<\|--|--\|>)\s+(.+?)(?:\s*:\s*(.+))?\s*$')
_RE_STEREO = re.compile(r"<<\s*(include|extend)\s*>>", re.IGNORECASE)

_RE_COND = re.compile(r"(?:condition\s*=\s*|触发条件[:：]\s*)(.+)$", re.IGNORECASE)
_RE_EXTPT = re.compile(r"(?:extension\s*point\s*[:：]\s*|扩展点[:：]\s*)(.+)$", re.IGNORECASE)
# [PATCH] 修复候选里常见的“关系行内联定义”：  ... ..> "名称" as ALIAS : include
_RE_INLINE_AS = re.compile(r'^\s*(".*?"|[^"\s].*?)\s+as\s+([A-Za-z_]\w*)\s*$')



def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def _norm_token(s: str) -> str:
    return s.strip().strip('"').strip()


def _extract_cond_extpt(text: str) -> Tuple[str, str]:
    cond = ""
    extpt = ""
    if not text:
        return cond, extpt

    # 触发条件
    m = _RE_COND.search(text)
    if m:
        cond = m.group(1).strip().strip('"')

    # 扩展点
    m2 = _RE_EXTPT.search(text)
    if m2:
        extpt = m2.group(1).strip().strip('"')

    return cond, extpt

# [PATCH] 将不规范 PlantUML 轻量修复为可渲染/可解析的形式
def normalize_usecase_plantuml(text: str) -> Tuple[str, List[str]]:
    """
    修复两类常见问题（最小侵入）：
    1) 关系行内联定义：  A ..> "X" as UCX : include
       =>  usecase "X" as UCX
           A ..> UCX : <<include>>
    2) : include / : extend => : <<include>> / : <<extend>>
    返回：(fixed_text, fixes)
    """
    fixes: List[str] = []
    lines = text.splitlines()

    defined_aliases: Set[str] = set()
    # 扫一遍收集已定义 alias（actor / usecase / (.. ) as ..）
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("'") or s.startswith("//"):
            continue
        m = _RE_ACTOR.match(ln)
        if m:
            alias = m.group(2) or _strip_quotes(m.group(1))
            defined_aliases.add(alias)
            continue
        m = _RE_UC_PAREN.match(ln)
        if m:
            alias = m.group(2) or m.group(1).strip()
            defined_aliases.add(alias)
            continue
        m = _RE_UC_KW.match(ln)
        if m:
            alias = m.group(2) or _strip_quotes(m.group(1))
            defined_aliases.add(alias)
            continue

    def _norm_label(lbl: str) -> str:
        if not lbl:
            return lbl
        # 已经有 <<include>>/<<extend>> 就不动
        if _RE_STEREO.search(lbl):
            return lbl
        low = lbl.lower().strip()
        if low == "include":
            fixes.append("normalize label: include -> <<include>>")
            return "<<include>>"
        if low == "extend":
            fixes.append("normalize label: extend -> <<extend>>")
            return "<<extend>>"
        # 形如 "include xxx"：只补 stereotype，后续文本保留为普通 label（PlantUML可接受）
        if low.startswith("include "):
            fixes.append("normalize label: include* -> <<include>> ...")
            return "<<include>> " + lbl[len("include "):].lstrip()
        if low.startswith("extend "):
            fixes.append("normalize label: extend* -> <<extend>> ...")
            return "<<extend>> " + lbl[len("extend "):].lstrip()
        return lbl

    pending_defs: List[str] = []
    new_lines: List[str] = []

    for ln in lines:
        mrel = _RE_REL.match(ln)
        if not mrel:
            new_lines.append(ln)
            continue

        left = _norm_token(mrel.group(1))
        arrow = mrel.group(2).strip()
        right = _norm_token(mrel.group(3))
        label = (mrel.group(4) or "").strip()

        def _extract_inline(token: str) -> Tuple[str, Optional[str], Optional[str]]:
            """
            token 可能是：  UC1  或  "删除图形" as UC4
            返回： (ref, name, alias)
            """
            m = _RE_INLINE_AS.match(token)
            if not m:
                return token, None, None
            name = _strip_quotes(m.group(1))
            alias = m.group(2)
            return alias, name, alias

        left_ref, left_name, left_alias = _extract_inline(left)
        right_ref, right_name, right_alias = _extract_inline(right)

        # 若关系两端出现内联定义，则补充 usecase 定义行
        for name, alias in ((left_name, left_alias), (right_name, right_alias)):
            if name and alias and alias not in defined_aliases:
                pending_defs.append(f'usecase "{name}" as {alias}')
                defined_aliases.add(alias)
                fixes.append(f"hoist inline usecase: {alias}='{name}'")

        label = _norm_label(label)

        if label:
            new_lines.append(f"{left_ref} {arrow} {right_ref} : {label}")
        else:
            new_lines.append(f"{left_ref} {arrow} {right_ref}")

    if not pending_defs:
        return text, fixes

    # 把补充的 usecase 定义插入到 @startuml 之后（最稳妥的最小插入点）
    out: List[str] = []
    inserted = False
    for ln in new_lines:
        out.append(ln)
        if not inserted and _RE_START.match(ln.strip()):
            out.extend(pending_defs)
            out.append("")  # 空行隔开
            inserted = True

    if not inserted:
        # 没有 @startuml，就直接前置
        out = pending_defs + [""] + out
        fixes.append("prepend missing @startuml context (defs hoisted on top)")

    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), fixes



def parse_plantuml_usecase_diagram(text: str, provenance: Dict[str, str] | None = None) -> DiagramIR:
    """
    支持至少：
      actor "User" as U
      (Create Shape) as UC1
      U --> UC1
      UC1 ..> UC2 : <<include>>
      UC3 ..> UC4 : <<extend>>
    并解析关系前的注释作为 justification：
      ' 因果说明：...
      UC3 ..> UC4 : <<extend>> condition=...
    """
    # [PATCH] 先把不规范候选修正为可解析结构，避免后续 merge/validate 出现脏节点
    text, _ = normalize_usecase_plantuml(text)

    provenance = provenance or {}

    alias_to_name: Dict[str, str] = {}
    actors: Set[str] = set()
    usecases: Set[str] = set()
    relations: List[Relation] = []

    lines = text.splitlines()
    in_block = False

    # 收集紧邻关系行前的注释（仅 ' 行）
    pending_just_lines: List[str] = []

    for idx, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            continue

        if _RE_START.match(stripped):
            in_block = True
            continue
        if _RE_END.match(stripped):
            in_block = False
            continue
        if not in_block and ("actor" in stripped or "usecase" in stripped or "(" in stripped or "--" in stripped or ".." in stripped):
            in_block = True

        # 支持两类注释：'（用于因果解释），//（普通注释丢弃）
        if stripped.startswith("//"):
            continue
        if stripped.startswith("'"):
            # 收集因果解释
            pending_just_lines.append(stripped.lstrip("'").strip())
            continue

        m = _RE_ACTOR.match(line)
        if m:
            name = _strip_quotes(m.group(1))
            alias = m.group(2) or name
            alias_to_name[alias] = name
            actors.add(name)
            pending_just_lines.clear()
            continue

        m = _RE_UC_PAREN.match(line)
        if m:
            name = m.group(1).strip()
            alias = m.group(2) or name
            alias_to_name[alias] = name
            usecases.add(name)
            pending_just_lines.clear()
            continue

        m = _RE_UC_KW.match(line)
        if m:
            name = _strip_quotes(m.group(1))
            alias = m.group(2) or name
            alias_to_name[alias] = name
            usecases.add(name)
            pending_just_lines.clear()
            continue

        m = _RE_REL.match(line)
        if m:
            left = _norm_token(m.group(1))
            arrow = m.group(2).strip()
            right = _norm_token(m.group(3))
            label = (m.group(4) or "").strip()

            src = alias_to_name.get(left, _strip_quotes(left))
            dst = alias_to_name.get(right, _strip_quotes(right))

            kind = "assoc"
            if arrow in ("<|--", "--|>"):
                kind = "generalize"
            else:
                sm = _RE_STEREO.search(label)
                if sm:
                    stereo = sm.group(1).lower()
                    kind = "include" if stereo == "include" else "extend"

            justification = " ".join([x for x in pending_just_lines if x])
            pending_just_lines.clear()

            # condition/extension_point 优先从 label 抽取，再从 justification 抽取
            cond1, extpt1 = _extract_cond_extpt(label)
            cond2, extpt2 = _extract_cond_extpt(justification)
            condition = cond1 or cond2
            extension_point = extpt1 or extpt2

            relations.append(
                Relation(
                    src=src,
                    kind=kind,  # type: ignore
                    dst=dst,
                    label=label,
                    raw=line,
                    line_no=idx,
                    justification=justification,
                    condition=condition,
                    extension_point=extension_point,
                    provenance=dict(provenance),
                )
            )
            continue

        # 解析失败提示行号
        if any(x in stripped for x in ["actor", "usecase", "--", "..", "(", ")"]):
            raise PlantUMLParseError("无法解析的 PlantUML 行", idx, line)

        pending_just_lines.clear()

    return DiagramIR(actors=actors, usecases=usecases, relations=relations, alias_to_name=alias_to_name)


def generate_plantuml(diagram: DiagramIR, title: str = "UseCase") -> str:
    """生成保守的 PlantUML（带 alias）。"""
    actor_alias: Dict[str, str] = {}
    uc_alias: Dict[str, str] = {}

    def _mk_alias(prefix: str, i: int) -> str:
        return f"{prefix}{i}"

    a_list = sorted(diagram.actors)
    u_list = sorted(diagram.usecases)

    for i, a in enumerate(a_list, start=1):
        actor_alias[a] = _mk_alias("A", i)
    for i, u in enumerate(u_list, start=1):
        uc_alias[u] = _mk_alias("UC", i)

    lines: List[str] = []
    lines.append("@startuml")
    lines.append(f"title {title}")
    lines.append("left to right direction")
    lines.append("skinparam shadowing false")
    lines.append("")

    for a in a_list:
        lines.append(f'actor "{a}" as {actor_alias[a]}')
    lines.append("")
    for u in u_list:
        lines.append(f"({u}) as {uc_alias[u]}")
    lines.append("")

    for r in diagram.relations:
        s = actor_alias.get(r.src, uc_alias.get(r.src, r.src))
        t = actor_alias.get(r.dst, uc_alias.get(r.dst, r.dst))
        if r.kind == "assoc":
            lines.append(f"{s} --> {t}")
        elif r.kind == "include":
            lines.append(f"{s} ..> {t} : <<include>>")
        elif r.kind == "extend":
            lines.append(f"{s} ..> {t} : <<extend>>")
        elif r.kind == "generalize":
            lines.append(f"{s} --|> {t}")

    lines.append("@enduml")
    return "\n".join(lines) + "\n"
