"""步骤3 分层抽样:来源组不可拆的多标签迭代分层 80/10/10,种子 42(spec §4.3)。

切分前先按清洗文本去重(合并来源组;同文异标进人工裁决,不出现在输出)。
scikit-multilearn IterativeStratification 作用于来源组(组标签 = 成员并集);
两阶段:先 90/10 切出 test,再把剩余 8/9 : 1/9 切 train/val(≈80/10/10)。

API 核对(2026-10-10;uv.lock 锁 scikit-multilearn==0.2.0。Context7 未收录
scikit-multilearn,仅有 scikit-learn(无此类)——改以锁版安装源码逐行核对 +
实测验证为准,结论如下):
- IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[a, b]):
  fold_i 分得全体样本的 a/b 份额;sklearn split() 首个二元组为 (fold_1, fold_0),
  即 share[0] 对应元组第 2 位。按库自带 iterative_train_test_split 的
  [test_size, 1-test_size] 惯例传 [切出份, 保留份],以 (保留, 切出) 解包。
- 0.2.0 的 random_state 不生效:_iter_test_indices 里 check_random_state 返回值
  被丢弃,平局/负证据抽签走全局 np.random.choice;且锁版 sklearn 对
  shuffle=False 直接拒绝 random_state 入参(ValueError)。→ 复现只能
  np.random.seed(seed),结束后恢复全局 RNG 状态以免外泄。
- order=1 即 Sechidis11 原始逐标签分层(库自带 helper 用 order=2,不采)。
"""

from __future__ import annotations

import hashlib

from training.labels import canonical_json, label_names


def dedupe_by_text(records):
    by_text: dict[str, list[dict]] = {}
    for r in records:
        by_text.setdefault(r["text"], []).append(r)
    merged, conflicts = [], []
    for text, rs in by_text.items():
        label_sets = {tuple(sorted(r["labels"])) for r in rs}
        if len(label_sets) > 1:
            conflicts.append({"text": text,
                              "sample_ids": [r["sample_id"] for r in rs],
                              "label_sets": sorted(map(list, label_sets))})
            continue
        head = dict(rs[0])
        head["source_group_id"] = "+".join(sorted({r["source_group_id"] for r in rs}))
        head["sample_id"] = "+".join(sorted(r["sample_id"] for r in rs)) \
            if len(rs) > 1 else rs[0]["sample_id"]
        merged.append(head)
    return merged, conflicts


def _group_labels(records):
    groups: dict[str, set] = {}
    for r in records:
        groups.setdefault(r["source_group_id"], set()).update(r["labels"])
    return groups


def stratified_split(records, seed: int = 42) -> dict[str, str]:
    import numpy as np
    from skmultilearn.model_selection import IterativeStratification

    names = label_names()
    gid_of = {r["sample_id"]: r["source_group_id"] for r in records}
    gids = sorted(set(gid_of.values()))
    glabels = _group_labels(records)
    X = np.array(range(len(gids))).reshape(-1, 1)
    y = np.array([[1 if n in glabels[g] else 0 for n in names] for g in gids])

    def one_split(X, y, cut_share):
        # 切出份 = fold_0,落在返回元组第 2 位(见模块 docstring API 核对)
        strat = IterativeStratification(
            n_splits=2, order=1,
            sample_distribution_per_fold=[cut_share, 1 - cut_share])
        return next(strat.split(X, y))

    # 0.2.0 抽签走全局 np.random(见 docstring)→ 此处播种,完事恢复全局状态
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        # 第一阶段:切出 test(10%)
        rest_idx, test_idx = one_split(X, y, 0.1)
        # 第二阶段:剩余 90% 里切 val(占全体 10% → 占剩余 1/9)
        tr_idx, val_idx = one_split(X[rest_idx], y[rest_idx], 1 / 9)
    finally:
        np.random.set_state(rng_state)

    group_split: dict[str, str] = {}
    for i in rest_idx[tr_idx]:
        group_split[gids[i]] = "train"
    for i in rest_idx[val_idx]:
        group_split[gids[i]] = "val"
    for i in test_idx:
        group_split[gids[i]] = "test"
    return {sid: group_split[g] for sid, g in gid_of.items()}


def freeze_manifest(records, assignment, tax_version, seed: int = 42) -> dict:
    entries = []
    for r in sorted(records, key=lambda r: r["sample_id"]):
        entries.append({
            "sample_id": r["sample_id"],
            "source_group_id": r["source_group_id"],
            "split": assignment[r["sample_id"]],
            "content_hash": hashlib.sha256(
                canonical_json({"text": r["text"], "labels": sorted(r["labels"])}
                               ).encode("utf-8")).hexdigest()})
    dataset_version = hashlib.sha256(
        canonical_json(entries).encode("utf-8")).hexdigest()
    return {"seed": seed, "taxonomy_version": tax_version,
            "dataset_version": dataset_version, "entries": entries}


def split_report(records, assignment) -> dict:
    names = label_names()
    total = len(records)
    ratios, support = {}, {n: {"train": 0, "val": 0, "test": 0} for n in names}
    multi = 0
    for r in records:
        split = assignment[r["sample_id"]]
        ratios[split] = ratios.get(split, 0) + 1
        if len(r["labels"]) > 1:
            multi += 1
        for lb in r["labels"]:
            support[lb][split] += 1
    return {"actual_ratios": {k: round(v / total, 4) for k, v in ratios.items()},
            "per_class_support": support,
            "multilabel_share": round(multi / total, 4)}
