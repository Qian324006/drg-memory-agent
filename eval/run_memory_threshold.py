"""记忆入库阈值评测：把「0.6 是拍脑袋定的」变成「0.6 有数据支撑」。

两种模式：
    python -m eval.run_memory_threshold              # 用已标注的抽取结果打分（默认，不花 API）
    python -m eval.run_memory_threshold --extractor llm   # 走真实 LLM 抽取（21 次 API 调用）

为什么默认不走 LLM：
    先把「评分函数 + 阈值」单独测清楚，才能知道问题出在抽取还是出在阈值。
    这是评测里最常被搞混的一件事。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from app.memory.models import MemoryItem, compute_score

from eval.metrics import memory_threshold_sweep

EVAL_DIR = Path(__file__).resolve().parent
CASES_FILE = EVAL_DIR / "cases_memory.jsonl"
REPORT_DIR = EVAL_DIR / "reports"

DEFAULT_THRESHOLD = 0.6  # 对应 app/main.py 里的 THRESHOLD


def load_cases() -> list[dict]:
    lines = CASES_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def score_with_gold(case: dict) -> float:
    """用标注好的抽取结果直接算分：只测评分函数与阈值，不受 LLM 波动影响。"""
    item = MemoryItem(
        content=case["content"],
        type=case["type"],
        durability=case["durability"],
        key=None,
        explicitness=case["explicitness"],
        user_specific=case["user_specific"],
        action_relevance=case["action_relevance"],
    )
    return compute_score(item)


def score_with_llm(cases: list[dict]) -> dict[str, float]:
    """走真实抽取器：整条链路的真实表现（含 LLM 抽错的代价）。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_deepseek import ChatDeepSeek
    import os

    from app.memory.models import EXTRACT_PROMPT, MemoryDecision

    llm = ChatDeepSeek(
        model="deepseek-v4-flash",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    extractor = llm.with_structured_output(MemoryDecision)

    scores: dict[str, float] = {}
    for case in cases:
        decision = extractor.invoke(
            [
                SystemMessage(content=EXTRACT_PROMPT),
                HumanMessage(content=f"对话内容：\n用户：{case['utterance']}"),
            ]
        )
        if not decision.store or not decision.memories:
            scores[case["id"]] = 0.0
            print(f"  {case['id']}  抽取器判定不存")
            continue
        best = max(compute_score(m) for m in decision.memories)
        scores[case["id"]] = best
        print(f"  {case['id']}  抽取 {len(decision.memories)} 条，最高分 {best:.3f}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="记忆入库阈值评测")
    parser.add_argument(
        "--extractor",
        choices=["gold", "llm"],
        default="gold",
        help="gold=用标注好的抽取结果（免费、可复现）；llm=走真实抽取器（会调用 API）",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cases = load_cases()

    if args.extractor == "llm":
        print(f"走真实 LLM 抽取，共 {len(cases)} 条，会调用 {len(cases)} 次 API ...\n")
        llm_scores = score_with_llm(cases)
    else:
        llm_scores = {}

    scored = []
    for case in cases:
        score = score_with_gold(case) if args.extractor == "gold" else llm_scores[case["id"]]
        scored.append({**case, "score": score})

    total_pos = sum(1 for c in scored if c["should_store"])

    # ---------- 1. 当前阈值 0.6 下的逐条判定 ----------
    print("=" * 92)
    print(f"1) 当前阈值 {DEFAULT_THRESHOLD} 下，每句话会不会被存进记忆库")
    print("=" * 92)
    print(f"{'id':<5}{'分数':>6}{'阈值判定':>9}{'正确答案':>9}{'':>3}{'话语'}")
    fp = fn = tp = tn = 0
    for case in scored:
        predicted = case["score"] >= DEFAULT_THRESHOLD
        correct = case["should_store"]
        if predicted and correct:
            tp += 1
            mark = "  "
        elif predicted and not correct:
            fp += 1
            mark = "误存"
        elif not predicted and correct:
            fn += 1
            mark = "漏存"
        else:
            tn += 1
            mark = "  "
        print(
            f"{case['id']:<5}{case['score']:>6.3f}"
            f"{'存' if predicted else '不存':>7}{'存' if correct else '不存':>9}"
            f"{mark:>4}  {case['utterance'][:34]}"
        )

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / total_pos if total_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print("-" * 92)
    print(
        f"阈值 {DEFAULT_THRESHOLD}：精确率 {precision:.1%} / 召回率 {recall:.1%} / F1 {f1:.3f}  "
        f"(TP={tp} FP={fp} FN={fn} TN={tn})"
    )

    # ---------- 2. 阈值扫描 ----------
    print()
    print("=" * 92)
    print("2) 阈值扫描：把阈值从 0.30 到 1.00 走一遍，看哪个点最优")
    print("=" * 92)
    sweep, best = memory_threshold_sweep(
        [{"id": c["id"], "score": c["score"], "should_store": c["should_store"]} for c in scored]
    )
    print(f"{'阈值':>6}{'精确率':>9}{'召回率':>9}{'F1':>8}{'误存':>6}{'漏存':>6}")
    for row in sweep:
        if abs((row["t"] * 100) % 4) < 0.01:
            print(
                f"{row['t']:>6.2f}{row['precision']:>9.1%}{row['recall']:>9.1%}"
                f"{row['f1']:>8.3f}{row['fp']:>6}{row['fn']:>6}"
            )

    print()
    print(f"F1 最优阈值 {best['t']:.2f}（F1={best['f1']:.3f}）；当前用 {DEFAULT_THRESHOLD}")
    if best["t"] > DEFAULT_THRESHOLD:
        print(f"  → 当前阈值偏松，调到 {best['t']:.2f} 能减少误存")
    elif best["t"] < DEFAULT_THRESHOLD:
        print(f"  → 当前阈值偏紧，降到 {best['t']:.2f} 能减少漏存")
    else:
        print("  → 当前阈值恰好落在最优点附近，说明 0.6 是合理的，只是之前缺数据支撑")

    # ---------- 3. 阈值救不了的情况 ----------
    print()
    print("=" * 92)
    print("3) 阈值救不了的情况（无论怎么调阈值都判错）")
    print("=" * 92)
    always_wrong = [
        c
        for c in scored
        if (c["should_store"] is False and c["score"] >= max(r["t"] for r in sweep))
        or (c["should_store"] is True and c["score"] < min(r["t"] for r in sweep))
    ]
    if always_wrong:
        for case in always_wrong:
            print(f"  {case['id']}  分数 {case['score']:.3f}  {case['note']}")
    else:
        print("  无（本批用例里没有被阈值完全放弃的样本）")
    print("  ↑ 这类样本说明：阈值只能在「分数」这个维度上做权衡，")
    print("    像敏感信息、临时状态这种语义判断，必须靠额外的规则或提示词拦截。")

    # ---------- 4. 存报告 ----------
    if not args.no_save:
        REPORT_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORT_DIR / f"memory_threshold_{stamp}.md"
        lines = [
            "# 记忆入库阈值评测报告",
            "",
            f"- 时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
            f"- 抽取方式：{args.extractor}（gold=标注结果，llm=真实抽取器）",
            f"- 用例数：{len(scored)}（该存 {total_pos} 条 / 不该存 {len(scored) - total_pos} 条）",
            f"- 当前阈值 {DEFAULT_THRESHOLD}：精确率 {precision:.1%} / 召回率 {recall:.1%} / F1 {f1:.3f}",
            f"- F1 最优阈值：{best['t']:.2f}（F1={best['f1']:.3f}）",
            "",
            "## 阈值扫描",
            "",
            "| 阈值 | 精确率 | 召回率 | F1 | 误存 | 漏存 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in sweep:
            if abs((row["t"] * 100) % 4) < 0.01:
                lines.append(
                    f"| {row['t']:.2f} | {row['precision']:.1%} | {row['recall']:.1%} "
                    f"| {row['f1']:.3f} | {row['fp']} | {row['fn']} |"
                )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n报告已写入 {path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
