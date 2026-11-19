"""检索评测：Recall@K / MRR / 拒答表现 / 相似度阈值扫描。

用法（在项目根目录 drg_memory_agent 下）：
    python -m eval.run_retrieval
    python -m eval.run_retrieval --k 3
    python -m eval.run_retrieval --no-save

为什么不需要人工判断答案对不对：
    每条「可答」用例都自带 gold_id（该问题对应的知识库文档 id），
    只要看这条文档有没有出现在检索结果前 K 名里，就能自动算 0/1，不用人读答案。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from app.knowledge.database import knowledge_db

from eval.metrics import mean, recall_at_k, reciprocal_rank, threshold_sweep

EVAL_DIR = Path(__file__).resolve().parent
CASES_FILE = EVAL_DIR / "cases_retrieval.jsonl"
REPORT_DIR = EVAL_DIR / "reports"

KINDS = {
    "answerable_verbatim": "可答·原文问法",
    "answerable_paraphrase": "可答·口语改写",
    "unanswerable_off_topic": "不可答·完全跑题",
    "unanswerable_near_domain": "不可答·同领域但库里没有",
}


def retrieve(query: str, k: int) -> tuple[list[str], list[float]]:
    """与 app 里 search_drg_knowledge 使用同一个 collection 和同一个 embedding 函数，
    区别只是额外取回文档 id 和分数，方便自动打分。"""
    vector = knowledge_db.embeddings.embed_query(query)
    result = knowledge_db._collection.query(query_embeddings=[vector], n_results=k)
    ids = result["ids"][0]
    distances = result["distances"][0]
    return ids, distances


def load_cases() -> list[dict]:
    lines = CASES_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="DRG 知识库检索评测")
    parser.add_argument("--k", type=int, default=5, help="检索条数，默认 5")
    parser.add_argument("--no-save", action="store_true", help="只打印，不写报告文件")
    args = parser.parse_args()

    cases = load_cases()
    rows: list[dict] = []

    print(f"共 {len(cases)} 条用例，检索 top-{args.k} ...\n")

    for case in cases:
        ids, distances = retrieve(case["question"], args.k)
        top1_sim = 1.0 - distances[0] if distances else 0.0
        rows.append(
            {
                **case,
                "ranked": ids,
                "distances": distances,
                "top1_sim": top1_sim,
                "hit1": recall_at_k(ids, case["gold_id"], 1),
                "hit3": recall_at_k(ids, case["gold_id"], 3),
                "hit5": recall_at_k(ids, case["gold_id"], 5),
                "rr": reciprocal_rank(ids, case["gold_id"]),
            }
        )

    # ---------- 1. 分组统计 ----------
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["kind"]].append(row)

    print("=" * 78)
    print("1) 分组指标")
    print("=" * 78)
    header = f"{'分组':<26}{'条数':>5}{'Hit@1':>8}{'Hit@3':>8}{'MRR':>8}{'top1相似度':>12}"
    print(header)

    for kind, items in grouped.items():
        hit1 = mean([r["hit1"] for r in items if r["hit1"] is not None])
        hit3 = mean([r["hit3"] for r in items if r["hit3"] is not None])
        mrr = mean([r["rr"] for r in items if r["rr"] is not None])
        sims = [r["top1_sim"] for r in items]
        label = KINDS.get(kind, kind)
        hit_text = f"{hit1:>8.1%}{hit3:>8.1%}{mrr:>8.3f}" if items[0]["gold_id"] else f"{'-':>8}{'-':>8}{'-':>8}"
        print(f"{label:<26}{len(items):>5}{hit_text}{mean(sims):>12.3f}")

    answerable = [r for r in rows if r["gold_id"]]
    print("-" * 78)
    print(
        f"{'可答问题合计':<26}{len(answerable):>5}"
        f"{mean([r['hit1'] for r in answerable]):>8.1%}"
        f"{mean([r['hit3'] for r in answerable]):>8.1%}"
        f"{mean([r['rr'] for r in answerable]):>8.3f}"
        f"{mean([r['top1_sim'] for r in answerable]):>12.3f}"
    )

    # ---------- 2. 当前实现的拒答能力 ----------
    unanswerable = [r for r in rows if not r["gold_id"]]
    print()
    print("=" * 78)
    print("2) 当前实现（search_drg_knowledge）的拒答表现")
    print("=" * 78)
    print("它的拒答条件是 `if not docs`，而 k=3 永远能返回 3 条文档，")
    print(f"所以对 {len(unanswerable)} 条「库里没有答案」的问题，实际拒答 0 条 → 拒答率 0.0%")

    # ---------- 3. 阈值扫描 ----------
    print()
    print("=" * 78)
    print("3) 如果改成「相似度低于阈值就拒答」，能救回多少？")
    print("=" * 78)

    sims_a = [r["top1_sim"] for r in answerable]
    sims_u_off = [r["top1_sim"] for r in rows if r["kind"] == "unanswerable_off_topic"]
    sims_u_near = [r["top1_sim"] for r in rows if r["kind"] == "unanswerable_near_domain"]

    sweep_rows, best_all = threshold_sweep(sims_a, sims_u_off + sims_u_near)
    _, best_off = threshold_sweep(sims_a, sims_u_off)

    print(f"{'阈值':>6}{'整体准确率':>12}{'可答放行率':>12}{'不可答拦截率':>14}")
    for row in sweep_rows:
        if abs((row["t"] * 100) % 4) < 0.01:  # 每 0.04 打一行，避免刷屏
            print(
                f"{row['t']:>6.2f}{row['acc']:>12.1%}"
                f"{row['recall_answerable']:>12.1%}{row['specificity']:>14.1%}"
            )

    sims_u = sims_u_off + sims_u_near
    print()
    print(f"基线（完全不拒答，全靠硬答）准确率：{len(answerable) / len(rows):.1%}")
    print(f"只看「完全跑题」：最优阈值 {best_off['t']:.2f}，准确率 {best_off['acc']:.1%}")
    print(f"算上「同领域但库里没有」：最优阈值 {best_all['t']:.2f}，准确率 {best_all['acc']:.1%}")
    print()
    print("为什么一个阈值就够了 / 它到底有多稳，看这两类分数的间隔：")
    print(f"  可答问题   top1 相似度：最小 {min(sims_a):.3f} / 均值 {mean(sims_a):.3f}")
    print(f"  不可答问题 top1 相似度：最大 {max(sims_u):.3f} / 均值 {mean(sims_u):.3f}")
    gap_lo, gap_hi = max(sims_u), min(sims_a)
    print(f"  可用阈值区间 [{gap_lo:.3f}, {gap_hi:.3f}]，宽度 {gap_hi - gap_lo:.3f}")
    print("  ↑ 宽度越窄，阈值越脆弱：真实用户换个说法就可能掉出安全区，")
    print("    所以阈值只是第一道闸，后面还要再加一层「让模型判断证据是否足够」的兜底。")

    # ---------- 4. 逐条明细 ----------
    print()
    print("=" * 78)
    print("4) 逐条明细（可答用例看 hit/排名，不可答用例看相似度）")
    print("=" * 78)
    for row in rows:
        if row["gold_id"]:
            rank = row["ranked"].index(row["gold_id"]) + 1 if row["gold_id"] in row["ranked"] else 0
            mark = "✓" if rank else "✗"
            print(f"{mark} {row['id']:<4}{row['kind'][11:]:<12}排名 {rank or '-':<3} sim={row['top1_sim']:.3f}  {row['question'][:40]}")
        else:
            print(f"  {row['id']:<4}{row['kind'][13:]:<12}          sim={row['top1_sim']:.3f}  {row['question'][:40]}")

    # ---------- 5. 存报告 ----------
    if not args.no_save:
        REPORT_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORT_DIR / f"retrieval_{stamp}.md"
        lines = [
            "# 检索评测报告",
            "",
            f"- 时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
            f"- 用例数：{len(cases)}（可答 {len(answerable)} / 不可答 {len(unanswerable)}）",
            f"- 检索条数 K：{args.k}",
            "",
            "## 分组指标",
            "",
            "| 分组 | 条数 | Hit@1 | Hit@3 | MRR | top1 相似度均值 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for kind, items in grouped.items():
            hit1 = mean([r["hit1"] for r in items if r["hit1"] is not None])
            hit3 = mean([r["hit3"] for r in items if r["hit3"] is not None])
            mrr = mean([r["rr"] for r in items if r["rr"] is not None])
            if items[0]["gold_id"]:
                lines.append(
                    f"| {KINDS.get(kind, kind)} | {len(items)} | {hit1:.1%} | {hit3:.1%} | {mrr:.3f} "
                    f"| {mean([r['top1_sim'] for r in items]):.3f} |"
                )
            else:
                lines.append(
                    f"| {KINDS.get(kind, kind)} | {len(items)} | - | - | - "
                    f"| {mean([r['top1_sim'] for r in items]):.3f} |"
                )
        lines += [
            "",
            "## 拒答与阈值",
            "",
            f"- 当前实现拒答率：0.0%（条件 `if not docs` 永远不成立）",
            f"- 基线（不拒答硬答）准确率：{len(answerable) / len(rows):.1%}",
            f"- 只看完全跑题的最优阈值：{best_off['t']:.2f}（准确率 {best_off['acc']:.1%}）",
            f"- 含同领域无答案的最优阈值：{best_all['t']:.2f}（准确率 {best_all['acc']:.1%}）",
            f"- 可答/不可答可用阈值区间：[{max(sims_u):.3f}, {min(sims_a):.3f}]"
            f"，宽度 {min(sims_a) - max(sims_u):.3f}（越窄越脆弱）",
            "",
            "| 阈值 | 整体准确率 | 可答放行率 | 不可答拦截率 |",
            "| --- | --- | --- | --- |",
        ]
        for row in sweep_rows:
            if abs((row["t"] * 100) % 4) < 0.01:
                lines.append(
                    f"| {row['t']:.2f} | {row['acc']:.1%} | {row['recall_answerable']:.1%} "
                    f"| {row['specificity']:.1%} |"
                )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n报告已写入 {path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
