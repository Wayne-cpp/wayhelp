"""步骤2前置:LLM 按术语表合成数据(每类 ~100 条,~20% 多诉求句)。

CLI: uv run --group train python -m training.prep_generate \
       --per-class 100 --multi-ratio 0.2 --out data/topic_classifier/raw_synthetic.jsonl
生成时指定标签组合(意图标签),合成记录 source_group_id=sample_id(各自成组);
切分前去重时才合并同文本组(spec §4)。
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from training.labels import label_names, load_taxonomy
from training.prep_records import DATA_DIR, make_chat_model, make_record, write_jsonl

# 17 类拼音排序键(手动冻结,与 sample_id 的「类拼音序」语义配套;无 pypinyin 依赖)
_PINYIN = {
    "退换货": "tuihuanhuo", "物流": "wuliu", "尺码": "chima", "发票": "fapiao",
    "质量问题": "zhiliangwenti", "运费": "yunfei", "优惠活动": "youhuihuodong",
    "价保": "jiabao", "支付": "zhifu", "订单修改": "dingdanxiugai",
    "库存补货": "kucunbuhuo", "商品信息": "shangpinxinxi", "保修维修": "baoxiuweixiu",
    "账号": "zhanghao", "会员积分": "huiyuanjifen", "评价": "pingjia", "其他": "qita",
}


def build_prompt(entry: dict, combos: list[list[str]], n: int) -> str:
    """entry: taxonomy 中一类;combos: 多诉求标签组合(含该类),长度 n 对应条数。"""
    lines = [f"你在为电商客服训练数据造句。类目「{entry['name']}」:{entry['boundary']}",
             f"请造 {n} 条用户提问短句(口语、带错别字空间、5-40 字),"
             "逐条输出 JSON 数组,元素为 {\"text\":..., \"labels\":[...]}。",
             "labels 只能来自给定组合,一句话字面提到几个诉求就打几个标签。"]
    return "\n".join(lines) + f"\n标签组合序列:{json.dumps(combos, ensure_ascii=False)}"


def parse_generated(text: str, entry_name: str) -> list[tuple[str, list[str]]]:
    """解析 LLM 输出的 JSON 数组;非法行丢弃(记返回条数供 CLI 对账)。"""
    start, end = text.find("["), text.rfind("]")
    items = json.loads(text[start:end + 1])
    out = []
    for it in items:
        t, lbs = (it.get("text") or "").strip(), it.get("labels") or []
        if t and isinstance(lbs, list) and lbs:
            out.append((t, list(lbs)))
    return out


def build_combos(entry_name: str, names: list[str], n: int, multi_ratio: float,
                 rng: random.Random) -> list[list[str]]:
    """单标签为主 + ~multi_ratio 双诉求组合;顺序打乱后逐条对应生成句。"""
    others = [x for x in names if x != entry_name]
    n_multi = round(n * multi_ratio)
    combos = [[entry_name] for _ in range(n - n_multi)]
    combos += [sorted([entry_name, rng.choice(others)]) for _ in range(n_multi)]
    rng.shuffle(combos)
    return combos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM 合成训练数据")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--multi-ratio", type=float, default=0.2)
    parser.add_argument("--out", default=str(DATA_DIR / "raw_synthetic.jsonl"))
    parser.add_argument("--seed", type=int, default=42)  # 全局种子冻结
    args = parser.parse_args(argv)

    from app.config import Settings
    taxonomy = load_taxonomy()
    names = label_names()
    ranks = {name: i for i, name in
             enumerate(sorted(names, key=lambda x: _PINYIN[x]))}
    rng = random.Random(args.seed)
    llm = make_chat_model(Settings())
    records = []
    for entry in taxonomy:
        combos = build_combos(entry["name"], names, args.per_class,
                              args.multi_ratio, rng)
        resp = llm.invoke([("user", build_prompt(entry, combos, args.per_class))])
        content = resp.content if isinstance(resp.content, str) else ""
        try:
            pairs = parse_generated(content, entry["name"])
        except (ValueError, json.JSONDecodeError):
            print(f"[generate] {entry['name']} 输出解析失败,该类跳过(可重跑补齐)",
                  file=sys.stderr)
            continue
        mismatch = sum(1 for i, (_, lbs) in enumerate(pairs)
                       if i < len(combos) and lbs != combos[i])
        if mismatch:
            print(f"[generate] {entry['name']}: 回显标签与指定组合不符 {mismatch} 条,"
                  f"以指定组合为准", file=sys.stderr)
        for i, (text, _) in enumerate(pairs):
            intended = combos[i] if i < len(combos) else [entry["name"]]
            sid = f"syn-{ranks[entry['name']]:02d}-{i:04d}"
            records.append(make_record(sid, "synthetic", sid, text, text,
                                       intended, list(intended)))
        print(f"[generate] {entry['name']}: 请求 {args.per_class} 条,"
              f"得 {len(pairs)} 条")
    write_jsonl(args.out, records)
    print(f"[generate] 共 {len(records)} 条 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
