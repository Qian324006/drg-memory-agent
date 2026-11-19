"""检索与记忆评测的指标函数。

设计原则：能自动算的绝不靠人眼看。
- 检索层：只要知道「这条问题对应库里哪条文档」，命中与否就能自动算，不用人判断答案对不对。
- 记忆层：只要知道「这句话该不该存」，阈值扫描就能自动算精确率/召回率。
"""

from __future__ import annotations


def recall_at_k(ranked: list[str], gold: str | None, k: int) -> float | None:
    """命中@k：gold 是否出现在前 k 名里。返回 0/1，无 gold 时返回 None。"""
    if gold is None:
        return None
    return 1.0 if gold in ranked[:k] else 0.0


def reciprocal_rank(ranked: list[str], gold: str | None) -> float | None:
    """MRR 的单条贡献：1/排名，没命中记 0。"""
    if gold is None:
        return None
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id == gold:
            return 1.0 / i
    return 0.0


def mean(values: list[float]) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def threshold_sweep(
    answerable_sims: list[float],
    unanswerable_sims: list[float],
    lo: float = 0.30,
    hi: float = 0.90,
    step: float = 0.02,
) -> tuple[list[dict], dict]:
    """扫描「相似度阈值」：低于阈值就当查不到、直接拒答。

    对每个阈值 t 统计：
      - 可答问题的放行率（recall_answerable）：分数 >= t 的比例，越高越好
      - 不可答问题的拦截率（specificity）：分数 <  t 的比例，越高越好
      - 整体准确率（acc）：两类都被判对的比例
    返回 (逐行明细, 最优阈值)。
    """
    rows: list[dict] = []
    best = {"t": None, "acc": -1.0}

    n_a, n_u = len(answerable_sims), len(unanswerable_sims)
    # 用整数步进再除，避免 0.30 + 0.02 + 0.02... 累积出 0.6000000000000003 这种数，
    # 否则恰好等于阈值的样本会时判对时判错。
    steps = int(round((hi - lo) / step)) + 1
    for i in range(steps):
        t = round(lo + i * step, 4)
        passed = sum(1 for s in answerable_sims if s >= t)      # 可答 -> 放行
        blocked = sum(1 for s in unanswerable_sims if s < t)     # 不可答 -> 拒答
        acc = (passed + blocked) / (n_a + n_u) if (n_a + n_u) else 0.0
        rows.append(
            {
                "t": round(t, 2),
                "acc": round(acc, 3),
                "recall_answerable": round(passed / n_a, 3) if n_a else 0.0,
                "specificity": round(blocked / n_u, 3) if n_u else 0.0,
            }
        )
        # 只在准确率严格更高时更新，避免同分时偏向高阈值
        if acc > best["acc"] + 1e-9:
            best = {"t": round(t, 2), "acc": round(acc, 3)}

    return rows, best


def memory_threshold_sweep(
    scored_cases: list[dict],
    lo: float = 0.30,
    hi: float = 1.00,
    step: float = 0.02,
) -> tuple[list[dict], dict]:
    """扫描「记忆入库阈值」：score >= t 就存。

    scored_cases 每项形如 {"id":..., "score": 0.92, "should_store": True}。
    返回 (逐行明细, F1 最优阈值)。
    """
    rows: list[dict] = []
    best = {"t": None, "f1": -1.0}

    total_pos = sum(1 for c in scored_cases if c["should_store"])
    # 同 threshold_sweep：整数步进，保证「恰好等于阈值」的样本判定稳定。
    steps = int(round((hi - lo) / step)) + 1
    for i in range(steps):
        t = round(lo + i * step, 4)
        tp = sum(1 for c in scored_cases if c["should_store"] and c["score"] >= t)
        fp = sum(1 for c in scored_cases if not c["should_store"] and c["score"] >= t)
        fn = total_pos - tp
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / total_pos if total_pos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "t": round(t, 2),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
        if f1 > best["f1"] + 1e-9:
            best = {"t": round(t, 2), "f1": round(f1, 3)}

    return rows, best
