"""评估集 loader(spec §4.5):CSV + GT 文法(+ AND;| 与 / 等价 OR)+ 固定奇偶分片。"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

BUCKETS = ("A_policy", "B_model", "C_colloquial", "D_absent", "E_multi")


@dataclass(frozen=True)
class EvalCase:
    id: str
    bucket: str
    query: str
    gt_groups: tuple[tuple[str, ...], ...]
    expect_points: tuple[str, ...]
    should_refuse: bool
    split: str  # "calibration" | "test"


def parse_gt(expr: str) -> tuple[tuple[str, ...], ...]:
    """文法:expression := or_group ("+" or_group)*;or_group := alias (("|" | "/") alias)*。
    空白剔除;空组/连续运算符/未知语法直接 ValueError。"""
    if not expr or not expr.strip():
        raise ValueError("空 GT 表达式")
    groups = []
    for g in expr.split("+"):
        raw = re.split(r"[|/]", g)
        aliases = tuple(a.strip() for a in raw if a.strip())
        if not aliases or len(aliases) != len(raw):
            raise ValueError(f"非法 GT 表达式片段: {g!r}")
        groups.append(aliases)
    return tuple(groups)


def covered_groups(retrieved_section_paths: list[str],
                   groups: tuple[tuple[str, ...], ...]) -> int:
    """召回块 section_path 包含组内任一别名即覆盖该组;一块可覆盖多组。"""
    return sum(1 for g in groups
               if any(any(alias in p for p in retrieved_section_paths) for alias in g))


def _split_of(case_id: str) -> str:
    m = re.search(r"(\d+)$", case_id)
    if not m:
        raise ValueError(f"case id 无数字后缀: {case_id!r}")
    return "calibration" if int(m.group(1)) % 2 == 1 else "test"


def load_compare_cases(path) -> list[EvalCase]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    cases: list[EvalCase] = []
    for row in rows:
        cid = row["id"].strip()
        bucket = row["桶(bucket)"].strip()
        query = row["问题(query)"].strip()
        expect_section = row["期望章节(expect_section)"].strip()
        points = tuple(p.strip()
                       for p in row["标准要点(expect_points)"].split("|") if p.strip())
        refuse = row["应拒答(should_refuse)"].strip() == "是"
        if bucket not in BUCKETS:
            raise ValueError(f"未知桶: {bucket!r}({cid})")
        gt = parse_gt(expect_section) if expect_section else ()
        cases.append(EvalCase(cid, bucket, query, gt, points, refuse, _split_of(cid)))
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case id 重复")
    for bucket in BUCKETS:
        items = [c for c in cases if c.bucket == bucket]
        assert len(items) == 60, f"{bucket} 需 60 条,实际 {len(items)}"
        assert sum(1 for c in items if c.split == "calibration") == 30
        assert sum(1 for c in items if c.split == "test") == 30
    for c in cases:
        if c.bucket == "D_absent":
            assert c.should_refuse and not c.gt_groups and not c.expect_points, c.id
        else:
            assert not c.should_refuse and c.gt_groups, c.id
    return cases
