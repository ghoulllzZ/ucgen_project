# ucgen/asp_reasoner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import logging
import random
import os
from pathlib import Path

from .merge import normalize_name

logger = logging.getLogger("ucgen")

try:
    import clingo  # type: ignore
except Exception:
    clingo = None


_ASP_DUMP_ENV = "UCGEN_ASP_DUMP_DIR"

def _dump_asp_lp(filename: str, program: Optional[str]) -> None:
    """
    若环境变量 UCGEN_ASP_DUMP_DIR 设置为某目录，则把 program 写盘为该目录下的 filename。
    用于复现：clingo <file.lp>
    """
    if not program:
        return
    out_dir = os.getenv(_ASP_DUMP_ENV, "").strip()
    if not out_dir:
        return
    try:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / filename).write_text(program, encoding="utf-8")
        logger.debug("Wrote ASP program: %s", str(p / filename))
    except Exception as e:
        logger.debug("Failed to dump ASP program (%s): %s", filename, e)


# 你可以按领域扩展：上位词 -> 下位词集合
DEFAULT_TAXONOMY: Dict[str, List[str]] = {
    # 图形编辑器常见as
    "图形": ["矩形", "圆形", "椭圆", "菱形", "直线", "多边形", "三角形"],
    "形状": ["矩形", "圆形", "椭圆", "菱形", "直线", "多边形", "三角形"],
    "图元": ["矩形", "圆形", "椭圆", "菱形", "直线", "多边形", "三角形"],
    # 尺寸/缩放相关
    "调整尺寸": ["调整宽度", "调整高度", "调整大小", "缩放"],
    "改变尺寸": ["改变宽度", "改变高度", "改变大小", "缩放"],
}

# 常见动词前缀（用于粗粒度“动词-宾语”拆分）
VERBS: List[str] = [
    "创建", "新建", "绘制",
    "删除", "移除",
    "移动", "拖动",
    "选择", "选中",
    "缩放", "调整", "改变",
    "切换",
    "保存", "导出", "导入", "打开", "关闭",
]


def clingo_available() -> bool:
    return clingo is not None


def _asp_escape(s: str) -> str:
    # clingo 字符串常量：用双引号包裹，内部转义 \ 和 "
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _split_verb_obj(name: str) -> Tuple[str, str]:
    n = name.strip()
    for v in VERBS:
        if n.startswith(v) and len(n) > len(v):
            return v, n[len(v):].strip()
    return "", n


def _infer_implies_pairs_from_name(usecases: Sequence[str], taxonomy: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
    """
    返回 (general, specific, reason)
    """
    implies: List[Tuple[str, str, str]] = []
    # 预处理：verb/obj
    parsed: Dict[str, Tuple[str, str]] = {u: _split_verb_obj(u) for u in usecases}

    # (1) taxonomy 规则：创建图形 -> 创建矩形/创建圆形...
    for g in usecases:
        gv, go = parsed[g]
        go_n = normalize_name(go)
        if not gv:
            continue
        for s in usecases:
            if s == g:
                continue
            sv, so = parsed[s]
            if sv != gv:
                continue
            so_n = normalize_name(so)
            # 若 general 的宾语是上位词，specific 的宾语在其下位词集合中
            for hyper, hypos in taxonomy.items():
                if normalize_name(hyper) == go_n:
                    for h in hypos:
                        if normalize_name(h) == so_n:
                            implies.append((g, s, f"taxonomy:{hyper}->{h}"))
                            break

    # (2) 前缀包含规则（更通用）：若 normalized(specific) 以 normalized(general) 开头且更长，认为 general 蕴含 specific
    # 例如：创建图形 -> 创建图形并设置颜色（或类似）
    for g in usecases:
        g_n = normalize_name(g)
        for s in usecases:
            if s == g:
                continue
            s_n = normalize_name(s)
            if g_n and s_n.startswith(g_n) and len(s_n) >= len(g_n) + 2:
                implies.append((g, s, "prefix_contains"))

    # 去重
    seen = set()
    out: List[Tuple[str, str, str]] = []
    for g, s, r in implies:
        key = (normalize_name(g), normalize_name(s), r)
        if key in seen:
            continue
        seen.add(key)
        out.append((g, s, r))
    return out


def _infer_implies_pairs_from_generalize_edges(generalize_edges: Iterable[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """
    generalize_edges: (child, parent)
    语义：parent 更抽象，child 更具体 => parent 蕴含 child
    """
    out = []
    for child, parent in generalize_edges:
        if parent and child and parent != child:
            out.append((parent, child, "generalize_edge"))
    return out


@dataclass
class AspPruneResult:
    drop_map: Dict[str, List[str]]   # specific -> [general...]
    used_asp: bool
    program: Optional[str] = None    # 便于调试（可选）


def asp_prune_usecases(
    usecases: Sequence[str],
    ambiguous_usecases: Set[str],
    generalize_edges: Iterable[Tuple[str, str]] = (),
    taxonomy: Optional[Dict[str, List[str]]] = None,
    keep_usecases: Optional[Set[str]] = None,
    seed: int = 7,
) -> AspPruneResult:
    """
    用 ASP（若可用）或 Python（降级）来过滤 “被更抽象用例蕴含的具体用例”。

    仅对 ambiguous_usecases 生效（避免误删高共识/明确用例）。
    keep_usecases 可强制保留（即使被蕴含也不删）。
    """
    taxonomy = taxonomy or DEFAULT_TAXONOMY
    keep_usecases = keep_usecases or set()

    uc_list = list(dict.fromkeys([u for u in usecases if u]))  # 保序去重

    implies1 = _infer_implies_pairs_from_name(uc_list, taxonomy)
    implies2 = _infer_implies_pairs_from_generalize_edges(generalize_edges)
    implies = implies1 + implies2

    if not implies:
        return AspPruneResult(drop_map={}, used_asp=False)

    # ------- clingo 分支 -------
    # if clingo_available():
    #     random.seed(seed)
    #
    #     facts: List[str] = []
    #     for u in uc_list:
    #         facts.append(f"uc({_asp_escape(u)}).")
    #     for u in ambiguous_usecases:
    #         if u in keep_usecases:
    #             continue
    #         facts.append(f"amb({_asp_escape(u)}).")
    #     for u in keep_usecases:
    #         facts.append(f"keep({_asp_escape(u)}).")
    #
    #     for g, s, reason in implies:
    #         facts.append(f"implies({_asp_escape(g)},{_asp_escape(s)},{_asp_escape(reason)}).")
    #
    #     program = "\n".join(facts) + "\n" + r"""

    # 生成 ASP program（无论 clingo 是否可用都写盘，便于复现）
    facts: List[str] = []
    for u in uc_list:
        facts.append(f"uc({_asp_escape(u)}).")
    for u in ambiguous_usecases:
        if u in keep_usecases:
            continue
        facts.append(f"amb({_asp_escape(u)}).")
    for u in keep_usecases:
        facts.append(f"keep({_asp_escape(u)}).")

    for g, s, reason in implies:
        facts.append(f"implies({_asp_escape(g)},{_asp_escape(s)},{_asp_escape(reason)}).")

    program = "\n".join(facts) + "\n" + r"""
% 只有 ambiguous 的 specific 才允许被删（避免过激）
drop(S,G,R) :- implies(G,S,R), uc(G), uc(S), amb(S), not keep(S).

dropped(S) :- drop(S,_,_).

#show drop/3.
#show dropped/1.
"""
    _dump_asp_lp("asp_prune.lp", program)

      # ------- clingo 分支 -------
    if clingo_available():
        random.seed(seed)

        try:
            ctl = clingo.Control(["--warn=none"])
            ctl.add("base", [], program)
            ctl.ground([("base", [])])

            drop_map: Dict[str, List[str]] = {}

            with ctl.solve(yield_=True) as handle:
                for model in handle:
                    atoms = model.symbols(shown=True)
                    for a in atoms:
                        if a.name == "drop" and len(a.arguments) >= 2:
                            s = a.arguments[0].string
                            g = a.arguments[1].string
                            if s not in drop_map:
                                drop_map[s] = []
                            drop_map[s].append(g)
                    break  # 取第一个模型即可

            # 去重
            for s in list(drop_map.keys()):
                drop_map[s] = sorted(list(set(drop_map[s])))

            # return AspPruneResult(drop_map=drop_map, used_asp=True, program=program)
            return AspPruneResult(drop_map=drop_map, used_asp=False, program=program)

        except Exception as e:
            logger.warning("ASP(clingo) prune failed, fallback to Python rules: %s", e)

    # ------- Python 降级分支 -------
    drop_map: Dict[str, List[str]] = {}
    for g, s, _r in implies:
        if s not in ambiguous_usecases:
            continue
        if s in keep_usecases:
            continue
        drop_map.setdefault(s, []).append(g)

    for s in list(drop_map.keys()):
        drop_map[s] = sorted(list(set(drop_map[s])))

    return AspPruneResult(drop_map=drop_map, used_asp=False)


# --- [PATCH] 从 justification 抽取 condition / extension_point ---
import re
from typing import Tuple

_RE_COND_1 = re.compile(r"(?:触发条件|条件)\s*[:：]\s*(.+)$")
_RE_COND_2 = re.compile(r"(当.+?时)")
_RE_COND_3 = re.compile(r"(如果.+?)(?:则|时|才|触发)")
_RE_COND_4 = re.compile(r"(在.+?时)")

_RE_EP_1 = re.compile(r"(?:扩展点|extension\s*point)\s*[:：]\s*(.+)$", re.IGNORECASE)
_RE_EP_2 = re.compile(r"(?:在|于)(.+?)(?:处|点|步骤|阶段)")

def extract_cond_extpt_from_justification(justification: str) -> Tuple[str, str]:
    """
    从 LLM 的 justification 文本里抽取触发条件/扩展点。
    尽量短、可读、可机读；抽不出来返回空字符串。
    """
    if not justification:
        return "", ""

    j = justification.strip()

    # condition
    cond = ""
    m = _RE_COND_1.search(j)
    if m:
        cond = m.group(1).strip()
    if not cond:
        m = _RE_COND_2.search(j)
        if m:
            cond = m.group(1).strip()
    if not cond:
        m = _RE_COND_3.search(j)
        if m:
            cond = m.group(1).strip()
    if not cond:
        m = _RE_COND_4.search(j)
        if m:
            cond = m.group(1).strip()

    # extension point
    ep = ""
    m = _RE_EP_1.search(j)
    if m:
        ep = m.group(1).strip()
    if not ep:
        # 只取短一点的片段，避免整个句子都进来
        m = _RE_EP_2.search(j)
        if m:
            ep = m.group(1).strip()
            # 太长就截断
            if len(ep) > 30:
                ep = ep[:30].strip()

    # 截断过长字段
    if len(cond) > 80:
        cond = cond[:80].strip()

    return cond, ep

# --- [PATCH] ASP: extend 因果约束检查 ---
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

try:
    import clingo  # type: ignore
except Exception:
    clingo = None

def _asp_escape(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'

@dataclass
class AspExtendCheckResult:
    bad_pairs: Set[str]              # key = "src|extend|dst"
    used_asp: bool
    program: Optional[str] = None    # 便于调试（可选）

def asp_check_extend_causality(
    extend_items: List[Dict[str, str]],
) -> AspExtendCheckResult:
    """
    输入：[{key, src, dst, condition, extension_point}, ...]
    输出：哪些 extend 不满足因果约束（缺少 condition 和 extension_point）
    - clingo 可用 => 用 ASP 求解
    - clingo 不可用 => Python 规则降级（保证可运行）
    """
    bad: Set[str] = set()

    # Python 降级（先算一遍，clingo 失败也能用）
    for it in extend_items:
        key = it.get("key", "")
        cond = (it.get("condition") or "").strip()
        ep = (it.get("extension_point") or "").strip()
        if key and (not cond) and (not ep):
            bad.add(key)

    # if clingo is None:
    #     return AspExtendCheckResult(bad_pairs=bad, used_asp=False, program=None)

    # 生成 ASP program（无论 clingo 是否可用都写盘，便于复现）
    facts: List[str] = []
    for it in extend_items:
        key = it.get("key", "")
        src = it.get("src", "")
        dst = it.get("dst", "")
        cond = (it.get("condition") or "").strip()
        ep = (it.get("extension_point") or "").strip()
        if not (key and src and dst):
            continue

        facts.append(f"ext({_asp_escape(key)},{_asp_escape(src)},{_asp_escape(dst)}).")
        if cond:
            facts.append(f"has_cond({_asp_escape(key)}).")
        if ep:
            facts.append(f"has_ep({_asp_escape(key)}).")

    program = "\n".join(facts) + "\n" + r"""
bad(K) :- ext(K,_,_), not has_cond(K), not has_ep(K).
#show bad/1.
"""
    _dump_asp_lp("asp_extend.lp", program)

    if clingo is None:
        return AspExtendCheckResult(bad_pairs=bad, used_asp=False, program=program)

    try:
        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], program)
        ctl.ground([("base", [])])

        bad2: Set[str] = set()
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                for a in model.symbols(shown=True):
                    if a.name == "bad" and len(a.arguments) == 1:
                        bad2.add(a.arguments[0].string)
                break

        return AspExtendCheckResult(bad_pairs=bad2, used_asp=True, program=program)

    except Exception:
        # clingo 出错则回退 Python 结果
        return AspExtendCheckResult(bad_pairs=bad, used_asp=False, program=program)
