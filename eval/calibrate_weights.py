"""权重校准：把「权重没依据」变成一个可回答的问题。

做四件事：
1. 用当前权重 + 闸门报告逐条分数，并拆解边界样本；
2. 在「权重和为 1」的单纯形上网格搜索 —— 但**只在训练集上搜**，留出集只用来验收一次；
3. 并列最优时，选「离专家先验最近」的那组（正则化到先验），而不是随便挑一组；
4. 敏感度分析：当前权重每个维度 ±0.05 / ±0.10，F1 掉不掉。

用法：
    python -m eval.calibrate_weights                        # 全量数据上的探索（会过拟合，看结构用）
    python -m eval.calibrate_weights --holdout 0.25 --human-only   # 正式协议：训练/验收分开
    python -m eval.calibrate_weights --step 0.1              # 数据少时用粗网格
"""

from __future__ import annotations

import argparse
import json
import random
import time
from itertools import product
from pathlib import Path

from app.memory.models import (
    ACTION_SCORE,
    DURABILITY_SCORE,
    EXPLICITNESS_SCORE,
    MemoryItem,
    passes_gate,
)

EVAL_DIR = Path(__file__).resolve().parent
CASES_FILE = EVAL_DIR / "cases_memory.jsonl"

DIMS = ("durability", "explicitness", "action_relevance", "user_specific")

# 专家先验：并列最优时的"回归点"，也是敏感度分析的基准
PRIOR = [0.40, 0.25, 0.20, 0.15]


def load_cases(human_only: bool) -> list[dict]:
    lines = CASES_FILE.read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]
    if human_only:
        cases = [c for c in cases if c.get("source") == "human"]
    return cases


def dimensions(case: dict) -> tuple[list[float], bool]:
    item = MemoryItem(
        content=case["content"],
        type=case["type"],
        durability=case["durability"],
        key=None,
        explicitness=case["explicitness"],
        user_specific=case["user_specific"],
        action_relevance=case["action_relevance"],
    )
    dims = [
        DURABILITY_SCORE[item.durability],
        EXPLICITNESS_SCORE[item.explicitness],
        ACTION_SCORE[item.action_relevance],
        1.0 if item.user_specific else 0.0,
    ]
    return dims, passes_gate(item)


def scores_for(weights: list[float], rows: list[dict]) -> list[float]:
    out = []
    for row in rows:
        dims, ok = row["dims"], row["gate"]
        out.append(round(sum(w * d for w, d in zip(weights, dims)), 3) if ok else 0.0)
    return out


def f1_at(scores: list[float], labels: list[bool], t: float) -> tuple[float, int, int, int]:
    tp = sum(1 for s, y in zip(scores, labels) if s >= t and y)
    fp = sum(1 for s, y in zip(scores, labels) if s >= t and not y)
    fn = sum(labels) - tp
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / sum(labels) if sum(labels) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return round(f1, 4), tp, fp, fn


def best_f1(scores: list[float], labels: list[bool]) -> tuple[float, float]:
    """在所有出现过的分数上找 F1 最优阈值（比等距扫描精确，不会漏边界）。"""
    candidates = sorted(set(scores))
    candidates = [candidates[0] - 1e-6] + candidates  # 全部不存
    best = (0.0, 0.0)
    for t in candidates:
        f1, _, _, _ = f1_at(scores, labels, t)
        if f1 > best[0] + 1e-9:
            best = (f1, round(t, 3))
    return best


def grid(step: float) -> list[list[float]]:
    axis = [round(i * step, 4) for i in range(int(round(1 / step)) + 1)]
    out = []
    for w1, w2, w3 in product(axis, repeat=3):
        w4 = round(1.0 - w1 - w2 - w3, 4)
        if w4 < -1e-9:
            continue
        out.append([w1, w2, w3, max(0.0, w4)])
    return out


def l1_to_prior(weights: list[float]) -> float:
    return sum(abs(w - p) for w, p in zip(weights, PRIOR))


def main() -> None:
    parser = argparse.ArgumentParser(description="记忆打分权重校准")
    parser.add_argument("--step", type=float, default=0.05, help="网格步长，默认 0.05")
    parser.add_argument("--holdout", type=float, default=0.0, help="留出集比例，如 0.25")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--human-only", action="store_true", help="只用人工标注的样本")
    args = parser.parse_args()

    cases = load_cases(args.human_only)
    if not cases:
        print("没有可用样本。先跑 python -m eval.label_tool 标一批。")
        return

    rows = []
    for case in cases:
        dims, gate = dimensions(case)
        rows.append({**case, "dims": dims, "gate": gate})
    labels = [r["should_store"] for r in rows]

    human = sum(1 for r in rows if r.get("source") == "human")
    ai = len(rows) - human
    print(f"样本 {len(rows)} 条：人工标注 {human} / AI 草稿 {ai}"
          f"｜该存 {sum(labels)} / 不该存 {len(labels) - sum(labels)}"
          f"｜被闸门拦下 {sum(1 for r in rows if not r['gate'])}")
    if args.human_only and human < 40:
        print(f"⚠️ 人工标注只有 {human} 条，低于 40 条时校准结果不可信（会并列一堆最优解）。")

    # ---------- 1. 当前权重 ----------
    cur_scores = scores_for(PRIOR, rows)
    cur_f1, cur_t = best_f1(cur_scores, labels)
    print()
    print("=" * 88)
    print("1) 当前权重（0.40 / 0.25 / 0.20 / 0.15）+ 闸门")
    print("=" * 88)
    print(f"{'id':<6}{'分数':>7}{'距0.60':>9}{'应存':>6}{'闸门':>6}  话语")
    for row, score in zip(rows, cur_scores):
        print(f"{row['id']:<6}{score:>7.3f}{score - 0.60:>+9.2f}"
              f"{'是' if row['should_store'] else '否':>6}"
              f"{'过' if row['gate'] else '拦':>6}  {row['utterance'][:24]}")
    print(f"\n最优阈值 {cur_t:.2f} 时 F1 = {cur_f1:.4f}")

    # ---------- 2. 边界样本拆解 ----------
    print()
    print("=" * 88)
    print("2) 边界样本拆解：分数是怎么被'地板'撑起来的")
    print("=" * 88)
    borderline = sorted(
        [r for r in rows if r["gate"]], key=lambda r: abs(scores_for(PRIOR, [r])[0] - 0.60)
    )[:2]
    for row in borderline:
        parts = [w * d for w, d in zip(PRIOR, row["dims"])]
        print(f"{row['id']}「{row['utterance'][:30]}」  应存={row['should_store']}")
        for name, w, d, p in zip(DIMS, PRIOR, row["dims"], parts):
            print(f"    {name:<18}w={w:.2f} × d={d:.2f} = {p:.3f}")
        print(f"    合计 {sum(parts):.3f}")

    # ---------- 3. 网格搜索（训练/验收分开） ----------
    print()
    print("=" * 88)
    if args.holdout > 0:
        print(f"3) 网格搜索（步长 {args.step}）｜训练 {1 - args.holdout:.0%} / 留出 {args.holdout:.0%}")
    else:
        print(f"3) 网格搜索（步长 {args.step}）｜⚠️ 未留出，下面的最优 F1 会偏乐观")
    print("=" * 88)

    if args.holdout > 0:
        idx = list(range(len(rows)))
        random.Random(args.seed).shuffle(idx)
        n_test = max(1, int(round(len(rows) * args.holdout)))
        test_idx = set(idx[:n_test])
        train = [r for i, r in enumerate(rows) if i not in test_idx]
        test = [r for i, r in enumerate(rows) if i in test_idx]
        y_train = [r["should_store"] for r in train]
        y_test = [r["should_store"] for r in test]
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            print("训练集或留出集只有一类样本，无法评估。增大数据量或减小 --holdout。")
            return
        print(f"训练 {len(train)} 条（该存 {sum(y_train)}）｜留出 {len(test)} 条（该存 {sum(y_test)}）")
    else:
        train, test = rows, rows
        y_train = y_test = labels
        print(f"全部 {len(train)} 条既当训练又当验收")

    start = time.time()
    searched = grid(args.step)
    results = []
    for weights in searched:
        f1, t = best_f1(scores_for(weights, train), y_train)
        results.append({"w": weights, "f1": f1, "t": t})
    results.sort(key=lambda r: (-r["f1"], l1_to_prior(r["w"])))

    max_f1 = results[0]["f1"]
    ties = [r for r in results if abs(r["f1"] - max_f1) < 1e-9]
    chosen = min(ties, key=lambda r: l1_to_prior(r["w"]))  # 并列时回归先验

    print(f"共评估 {len(searched)} 组，耗时 {time.time() - start:.1f}s")
    print(f"训练集最优 F1 = {max_f1:.4f}，并列 {len(ties)} 组（{len(ties) / len(searched):.1%}）")
    print(f"按「离专家先验最近」选中的权重：{['%.2f' % w for w in chosen['w']]}（阈值 {chosen['t']:.2f}）")

    test_scores = scores_for(chosen["w"], test)
    test_f1, tp, fp, fn = f1_at(test_scores, y_test, chosen["t"])
    cur_test_f1, _, _, _ = f1_at(scores_for(PRIOR, test), y_test, 0.60)
    print()
    print(f"{'':<22}{'训练集 F1':>12}{'留出集 F1':>12}{'差距':>9}")
    print(f"{'当前权重 @0.60':<22}{cur_f1:>12.4f}{cur_test_f1:>12.4f}{cur_test_f1 - cur_f1:>9.4f}")
    print(f"{'校准权重 @最优阈值':<22}{max_f1:>12.4f}{test_f1:>12.4f}{test_f1 - max_f1:>9.4f}")
    print(f"留出集混淆矩阵：TP={tp} FP={fp} FN={fn}")
    if max_f1 - test_f1 > 0.10:
        print("⚠️ 训练/留出差距 > 0.10：过拟合。样本还不够，别把校准权重写进代码。")
    elif test_f1 < cur_test_f1:
        print("⚠️ 校准权重在留出集上还不如当前权重 —— 说明当前先验已经不错，别改。")

    print()
    print("怎么用这个结果（决定权在你）：")
    print("  1. 留出集 F1 明显高于当前权重 + 差距 < 0.10 → 可以把权重换掉，并把报告路径写进代码注释；")
    print("  2. 两者差不多 → 保持先验，面试时讲「我校准过，数据显示先验已经接近最优」，这比换数字更有说服力；")
    print("  3. 差距 > 0.10 → 样本不够，继续标，别改代码。")

    # ---------- 4. 敏感度 ----------
    print()
    print("=" * 88)
    print("4) 敏感度：当前权重每个维度 ±0.05 / ±0.10，F1 会不会掉")
    print("=" * 88)
    print(f"{'维度':<18}{'-0.10':>9}{'-0.05':>9}{'基准':>9}{'+0.05':>9}{'+0.10':>9}")
    for i, name in enumerate(DIMS):
        cells = []
        for delta in (-0.10, -0.05, 0.0, 0.05, 0.10):
            weights = PRIOR[:]
            weights[i] = max(0.0, weights[i] + delta)
            rest = [j for j in range(4) if j != i]
            rest_sum = sum(PRIOR[j] for j in rest)
            target = 1.0 - weights[i]
            for j in rest:
                weights[j] = round(PRIOR[j] / rest_sum * target, 4) if rest_sum else 0.0
            f1, _ = best_f1(scores_for(weights, rows), labels)
            cells.append(f1)
        print(f"{name:<18}" + "".join(f"{c:>9.3f}" for c in cells)
              + ("   ← 掉分" if min(cells) < cur_f1 - 1e-9 else "   稳定"))


if __name__ == "__main__":
    main()
