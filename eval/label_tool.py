"""交互式标注工具：把「扩标注」变成 50–70 分钟能干完的活。

流程（每一步都有理由）：
    1. 显示候选话语；
    2. 让 LLM 抽取器给出结构化字段（type / durability / ...）—— 这两步是「机器先做」；
    3. **你先判「该存 / 不该存」，判完才显示分数** —— 避免被分数锚定（先看分数再判，等于在给现有权重投票）；
    4. 字段抽错了按 e 改；5. 追加写入 cases_memory.jsonl，自动跳过已标的。

用法：
    python -m eval.label_tool                 # 继续标注（从第一个未标的开始）
    python -m eval.label_tool --limit 20      # 这次只标 20 条
    python -m eval.label_tool --sample 20 --blind   # 一致性自检：盲标已标过的 20 条
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
CANDIDATES_FILE = EVAL_DIR / "candidates.jsonl"
CASES_FILE = EVAL_DIR / "cases_memory.jsonl"
CONSISTENCY_FILE = EVAL_DIR / "consistency.jsonl"

# 与 app/main.py 里的 THRESHOLD 保持一致；只用于「标完给你看管线会怎么判」
DEFAULT_THRESHOLD = 0.6

TYPE_CHOICES = ["semantic", "procedural", "episodic"]
DURABILITY_CHOICES = ["permanent", "long_term", "temporary", "one_off"]
EXPLICITNESS_CHOICES = ["explicit", "inferred"]
ACTION_CHOICES = ["high", "medium", "low"]

# 抽取器判定「不存」时，用这组字段代表「无可存内容」：分数一定是 0（闸门拦住）
EMPTY_EXTRACTION = {
    "content": "用户未提供可存事实",
    "type": "semantic",
    "durability": "one_off",
    "explicitness": "inferred",
    "user_specific": False,
    "action_relevance": "low",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_extractor():
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_deepseek import ChatDeepSeek

    from app.config import ENV_FILE
    from app.memory.models import EXTRACT_PROMPT, MemoryDecision

    load_dotenv(ENV_FILE, override=True)
    llm = ChatDeepSeek(
        model="deepseek-v4-flash",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    extractor = llm.with_structured_output(MemoryDecision)

    def run(utterance: str) -> tuple[dict, bool]:
        decision = extractor.invoke(
            [
                SystemMessage(content=EXTRACT_PROMPT),
                HumanMessage(content=f"对话内容：\n用户：{utterance}"),
            ]
        )
        if not decision.store or not decision.memories:
            return dict(EMPTY_EXTRACTION), False
        m = decision.memories[0]
        return (
            {
                "content": m.content,
                "type": m.type,
                "durability": m.durability,
                "explicitness": m.explicitness,
                "user_specific": m.user_specific,
                "action_relevance": m.action_relevance,
            },
            True,
        )

    return run


def show_fields(fields: dict) -> None:
    print(
        f"   抽取: {fields['type']:<10} {fields['durability']:<9} {fields['explicitness']:<9} "
        f"本人={'是' if fields['user_specific'] else '否'} 影响={fields['action_relevance']}"
    )
    print(f"   内容: {fields['content']}")


def edit_fields(fields: dict) -> dict:
    """逐项确认，直接回车保持原值。"""
    out = dict(fields)

    def ask(prompt: str, current, choices=None):
        hint = f" [{'/'.join(choices)}]" if choices else ""
        raw = input(f"     {prompt}{hint}（回车={current}）: ").strip()
        if not raw:
            return current
        if choices and raw not in choices:
            print(f"      ⚠️ 只接受 {'/'.join(choices)}，保持原值")
            return current
        return raw

    out["content"] = ask("content", out["content"])
    out["type"] = ask("type", out["type"], TYPE_CHOICES)
    out["durability"] = ask("durability", out["durability"], DURABILITY_CHOICES)
    out["explicitness"] = ask("explicitness", out["explicitness"], EXPLICITNESS_CHOICES)
    raw = input(f"     user_specific（y/n，回车={'y' if out['user_specific'] else 'n'}）: ").strip().lower()
    if raw in ("y", "n"):
        out["user_specific"] = raw == "y"
    out["action_relevance"] = ask("action_relevance", out["action_relevance"], ACTION_CHOICES)
    return out


def score_of(fields: dict) -> float:
    from app.memory.models import MemoryItem, compute_score

    return compute_score(MemoryItem(key=None, **fields))


def label_new(args) -> None:
    from app.memory.models import MemoryItem  # noqa: F401  (保证 app 可导入，报错更早)

    candidates = read_jsonl(CANDIDATES_FILE)
    labeled = read_jsonl(CASES_FILE)
    done_ids = {c["id"] for c in labeled}
    todo = [c for c in candidates if c["id"] not in done_ids]
    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        print(f"候选句已全部标完（{len(labeled)} 条）。想加新句子就编辑 eval/candidates.jsonl。")
        return

    print(f"已标 {len(labeled)} 条，本次待标 {len(todo)} 条。")
    print("指令：[y] 该存  [n] 不该存  [e] 改字段  [s] 跳过  [q] 退出\n")

    extract = build_extractor()
    counts = {"y": 0, "n": 0, "s": 0, "agree": 0, "total": 0}
    try:
        for i, cand in enumerate(todo, start=1):
            print("=" * 74)
            print(f"[{i}/{len(todo)}] {cand['id']}  分层：{cand['stratum']}")
            print(f"  「{cand['utterance']}」")

            fields, extractor_store = extract(cand["utterance"])
            if not extractor_store:
                print("   抽取: 判定「无需存储」（无 memories）")
            else:
                show_fields(fields)

            action = input("  判定 [y/n/e/s/q]: ").strip().lower()
            while action not in ("y", "n", "e", "s", "q"):
                action = input("  请输入 y / n / e / s / q: ").strip().lower()

            if action == "q":
                print("\n已退出。")
                break
            if action == "s":
                counts["s"] += 1
                continue
            if action == "e":
                fields = edit_fields(fields)

            should_store = action == "y"
            score = score_of(fields)
            pipeline_store = score >= DEFAULT_THRESHOLD
            agree = pipeline_store == should_store
            if agree:
                counts["agree"] += 1
            counts["total"] += 1
            counts[action] += 1

            print(
                f"   → 分数 {score:.3f}，阈值 {DEFAULT_THRESHOLD} 下会「{'存' if pipeline_store else '不存'}」"
                f"  {'✓ 一致' if agree else '✗ 不一致（这正是需要校准的信号）'}"
            )

            append_jsonl(
                CASES_FILE,
                {
                    "id": cand["id"],
                    "stratum": cand["stratum"],
                    "utterance": cand["utterance"],
                    "should_store": should_store,
                    **fields,
                    "extractor_store": extractor_store,
                    "source": "human",
                    "labeled_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
    except KeyboardInterrupt:
        print("\n已中断，本次已标的都保存了。")

    print()
    print("=" * 74)
    print(f"本次：存 {counts['y']} / 不存 {counts['n']} / 跳过 {counts['s']}")
    if counts["total"]:
        print(f"与当前管线一致率 {counts['agree'] / counts['total']:.1%}"
              f"（{counts['agree']}/{counts['total']}）")
    print(f"cases_memory.jsonl 现在共 {len(read_jsonl(CASES_FILE))} 条")
    print("\n下一步：python -m eval.calibrate_weights --holdout 0.25 --human-only")


def label_blind(args) -> None:
    """一致性自检：盲标已经标过的句子，比较两次是否一致。"""
    labeled = [c for c in read_jsonl(CASES_FILE) if c.get("source") == "human"]
    if len(labeled) < 10:
        print(f"人工标注只有 {len(labeled)} 条，先标到 20 条以上再做一致性检查。")
        return

    random.seed(args.seed)
    sample = random.sample(labeled, min(args.sample, len(labeled)))
    print(f"盲标 {len(sample)} 条（不显示你上次的答案）。指令：[y] 该存  [n] 不该存  [q] 退出\n")

    agree = 0
    compared = 0
    try:
        for i, case in enumerate(sample, start=1):
            print("=" * 74)
            print(f"[{i}/{len(sample)}] {case['id']}")
            print(f"  「{case['utterance']}」")
            action = input("  判定 [y/n/q]: ").strip().lower()
            while action not in ("y", "n", "q"):
                action = input("  请输入 y / n / q: ").strip().lower()
            if action == "q":
                break
            again = action == "y"
            same = again == case["should_store"]
            agree += int(same)
            compared += 1
            print(f"   上次：{'存' if case['should_store'] else '不存'} | 这次：{'存' if again else '不存'}"
                  f" | {'✓' if same else '✗ 争议'}")
            append_jsonl(
                CONSISTENCY_FILE,
                {
                    "id": case["id"],
                    "first": case["should_store"],
                    "second": again,
                    "same": same,
                    "at": datetime.now().isoformat(timespec="seconds"),
                },
            )
    except KeyboardInterrupt:
        print("\n已中断。")

    if compared:
        rate = agree / compared
        print()
        print("=" * 74)
        print(f"一致率 {rate:.1%}（{agree}/{compared}）")
        if rate >= 0.85:
            print("≥ 85%：标签可用，可以去做校准了。")
        else:
            print("< 85%：不是你的错，是判定标准还有模糊地带。")
            print("回去把争议最大的几条写进 eval/LABELING_GUIDE.md 的「争议例」，再重标。")


def main() -> None:
    parser = argparse.ArgumentParser(description="记忆标注工具")
    parser.add_argument("--limit", type=int, default=0, help="本次最多标几条")
    parser.add_argument("--sample", type=int, default=20, help="--blind 模式的抽样条数")
    parser.add_argument("--blind", action="store_true", help="一致性自检模式")
    parser.add_argument("--seed", type=int, default=0, help="--blind 抽样随机种子")
    args = parser.parse_args()

    if args.blind:
        label_blind(args)
    else:
        label_new(args)


if __name__ == "__main__":
    main()
