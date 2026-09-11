"""Markdown 结构感知切分(spec §5)。纯函数,无 IO,单测主力。"""

from dataclasses import dataclass, field

import re


class ChunkingError(ValueError):
    def __init__(self, message: str, source: str, line_no: int | None = None):
        self.source = source
        self.line_no = line_no
        where = f"{source}:{line_no}" if line_no is not None else source
        super().__init__(f"{where}: {message}")


@dataclass(frozen=True)
class Chunk:
    category: str
    questions: str
    answer: str
    section_path: str | None
    content_type: str
    is_key_clause: bool


_KEY_CLAUSE_WORDS = ("不支持", "不予", "必须", "扣除", "逾期", "无效")
_SENTENCE_END = "。！？!?"
_HEADING_RE = re.compile(r"^(#{1,3})\s+(\S.*?)\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])")


def _parse_frontmatter(text: str, source: str) -> tuple[str, str]:
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise ChunkingError("缺少 frontmatter(首行须为 ---)", source)
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise ChunkingError("frontmatter 未闭合", source, 1)
    meta = {}
    for ln in lines[1:end]:
        if ":" in ln:
            k, _, v = ln.partition(":")
            meta[k.strip()] = v.strip()
    ctype = meta.get("type", "")
    if ctype not in ("faq", "policy", "manual"):
        raise ChunkingError(f"frontmatter type 缺失或未知: {ctype!r}", source, 1)
    return ctype, "\n".join(lines[end + 1:])


@dataclass
class _Block:
    kind: str      # "para" | "table"
    text: str
    line_no: int   # 首行在传入 text 中的行号(1 起)


@dataclass
class _Section:
    stack: list[str]
    blocks: list[_Block] = field(default_factory=list)


def _parse_sections(body: str, source: str, split_levels: set[int]) -> tuple[str, list[_Section]]:
    """逐行解析。返回 (H1 文档标题, sections)。不在 split_levels 的标题行当正文。"""
    title: str | None = None
    sections: list[_Section] = []
    preamble: list[_Block] = []
    stack: list[str] = []
    cur: _Section | None = None
    para_lines: list[str] = []
    para_start = 0
    table_lines: list[str] = []
    table_start = 0

    def flush_para():
        nonlocal para_lines
        text = "\n".join(para_lines).strip() if para_lines else ""
        if text:
            (cur.blocks if cur else preamble).append(_Block("para", text, para_start))
        para_lines = []

    def flush_table():
        nonlocal table_lines
        if table_lines:
            (cur.blocks if cur else preamble).append(
                _Block("table", "\n".join(table_lines), table_start))
            table_lines = []

    for idx, line in enumerate(body.split("\n"), start=1):
        m = _HEADING_RE.match(line)
        if m and len(m.group(1)) in split_levels:
            flush_para()
            flush_table()
            level, heading = len(m.group(1)), m.group(2)
            stack = stack[: level - 1] + [heading]
            if level == 1 and title is None:
                title = heading
            cur = _Section(stack=list(stack))
            sections.append(cur)
        elif line.strip().startswith("|"):
            flush_para()
            if not table_lines:
                table_start = idx
            table_lines.append(line.rstrip())
        elif not line.strip():
            flush_para()
            flush_table()
        else:
            flush_table()
            if not para_lines:
                para_start = idx
            para_lines.append(line.rstrip())
    flush_para()
    flush_table()
    if title is None:
        raise ChunkingError("缺少 H1 文档标题", source, 1)
    # 前言并入第一个可产出知识块的 section
    if preamble:
        for sec in sections:
            if any(b.text.strip() for b in sec.blocks):
                sec.blocks = preamble + sec.blocks
                break
        else:
            sections.insert(0, _Section(stack=[title], blocks=preamble))
    return title, sections


def _split_sentences(text: str) -> list[str]:
    return [p for p in _SENTENCE_SPLIT_RE.split(text) if p]


def _hard_cut(text: str, max_chars: int) -> list[str]:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _para_units(text: str, max_chars: int) -> list[str]:
    """段落 → 可装入单元:整段 ≤ max 则整段;否则按句;单句超限硬切。"""
    if len(text) <= max_chars:
        return [text]
    units: list[str] = []
    for sent in _split_sentences(text):
        if len(sent) <= max_chars:
            units.append(sent)
        else:
            units.extend(_hard_cut(sent, max_chars))
    if not units:  # 无任何句末标点:整段视为一句
        units = _hard_cut(text, max_chars)
    return [u for u in units if u]


def _table_chunks(text: str, max_chars: int, source: str, base_line_no: int) -> list[str]:
    lines = text.split("\n")
    if len(lines) < 2:
        raise ChunkingError("表格缺少表头/分隔行", source, base_line_no)
    header = "\n".join(lines[:2])
    if len(header) > max_chars:
        raise ChunkingError("表头+分隔行本身超限,请缩短表头", source, base_line_no)
    chunks: list[str] = []
    cur = header
    for i, row in enumerate(lines[2:], start=3):
        if len(header) + 1 + len(row) > max_chars:
            raise ChunkingError("表头+单个数据行即超限,请缩短该行或拆分源表格",
                                source, base_line_no + i - 1)
        if len(cur) + 1 + len(row) > max_chars:
            chunks.append(cur)
            cur = header
        cur = cur + "\n" + row
    chunks.append(cur)
    return chunks


def _sentence_suffix(text: str, budget: int) -> str:
    """末尾 ≤ budget 的完整句后缀;块尾不是句末标点(硬切尾)则无重叠。"""
    if budget <= 0 or not text or text[-1] not in _SENTENCE_END:
        return ""
    best = ""
    for piece in reversed(_split_sentences(text)):  # 从末尾往前累积完整句
        candidate = piece + best
        if len(candidate) <= budget:
            best = candidate
        else:
            break
    return best.lstrip("\n")  # 跨段块的句片段带换行前缀,不属于句内容


def _emit_answers(blocks: list[_Block], max_chars: int, overlap: int,
                  source: str) -> list[str]:
    """blocks → 有序 answer 列表;文本块打包+完整句重叠,表格块独立成块。"""
    answers: list[tuple[str, str]] = []  # (kind, text)
    cur_text = ""
    cur_capacity = max_chars  # section 首块不预留重叠预算

    def close_text():
        nonlocal cur_text, cur_capacity
        if cur_text:
            answers.append(("text", cur_text))
            cur_text = ""
            cur_capacity = max_chars - (overlap + 1 if overlap > 0 else 0)

    for blk in blocks:
        if blk.kind == "table":
            close_text()
            for piece in _table_chunks(blk.text, max_chars, source, blk.line_no):
                answers.append(("table", piece))
            cur_capacity = max_chars - (overlap + 1 if overlap > 0 else 0)
            continue
        for j, unit in enumerate(_para_units(blk.text, max_chars)):
            # 段落之间用换行分隔;同段拆出的句单元直接拼接
            joiner = "\n" if cur_text and j == 0 else ""
            if cur_text and len(cur_text) + len(joiner) + len(unit) > cur_capacity:
                close_text()
                joiner = ""
            cur_text = cur_text + joiner + unit if cur_text else unit
    close_text()

    out: list[str] = []
    for i, (kind, text) in enumerate(answers):
        if kind == "text" and i > 0 and answers[i - 1][0] == "text":
            suffix = _sentence_suffix(out[-1], overlap)
            # 与下一完整句冲突(如硬切块满载)时取消重叠,保持 answer 不超限
            if suffix and len(suffix) + 1 + len(text) <= max_chars:
                text = suffix + "\n" + text
        out.append(text)
    return [t for t in out if t.strip()]


def chunk_document(text: str, *, source: str, max_chars: int,
                   overlap_chars: int) -> list[Chunk]:
    text = text.replace("\r\n", "\n")
    ctype, body = _parse_frontmatter(text, source)
    # faq: H1 仅作文档标题、## 产块、### 留正文;policy/manual: 三级都切
    split_levels = {1, 2} if ctype == "faq" else {1, 2, 3}
    title, sections = _parse_sections(body, source, split_levels)
    chunks: list[Chunk] = []
    pending_prefix: list[_Block] = []  # faq 下 H1 直挂正文,并入下一问答块
    for sec in sections:
        blocks = [b for b in sec.blocks if b.text.strip()]
        if ctype == "faq" and len(sec.stack) < 2:
            pending_prefix.extend(blocks)
            continue
        if pending_prefix:
            blocks = pending_prefix + blocks
            pending_prefix = []
        if not blocks:
            continue
        questions = sec.stack[-1]
        if ctype == "faq" or len(sec.stack) == 1:
            category = title
        else:
            category = " > ".join(sec.stack[:-1])
        for answer in _emit_answers(blocks, max_chars, overlap_chars, source):
            chunks.append(Chunk(
                category=category, questions=questions, answer=answer,
                section_path=" > ".join(sec.stack), content_type=ctype,
                is_key_clause=any(w in answer for w in _KEY_CLAUSE_WORDS),
            ))
    if not chunks:
        raise ChunkingError("整个文档无可入库正文", source)
    return chunks
