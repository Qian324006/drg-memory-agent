"""检查记忆库里还有哪些 v1 老数据需要补字段。

跑法（在 drg_memory_agent 目录下）：
    python -m scripts.audit_memory_v1

它是只读的：只统计、只打印，不改任何数据。

为什么要这个脚本：v2 加的两个维度在读取时都有兜底（缺 subject 当 user、缺
valid_until 当永不过期），所以老数据不会让程序崩。但"不崩"不等于"对"——
兜底的方向恰好是两种最糟的默认：
  - 缺 valid_until 的 temporary 记忆会永远不过期（本该消失的旧状态一直污染检索）
  - 缺 subject 的他人记忆会被当成用户自己的事（正是要防的幻觉）

所以要把这两类单独找出来看，而不是指望默认值。
"""

import sys

from app.memory.store import SCHEMA_VERSION, MemoryStore


def main() -> int:
    store = MemoryStore()
    col = store.vector_store._collection
    total = col.count()

    got = col.get(include=["metadatas", "documents"])
    ids = got["ids"]
    metas = got["metadatas"] or [{} for _ in ids]
    docs = got["documents"] or ["" for _ in ids]

    migrated = [i for i, md in zip(ids, metas) if md.get("schema_version") == SCHEMA_VERSION]
    legacy = [(i, md, d) for i, md, d in zip(ids, metas, docs) if md.get("schema_version") != SCHEMA_VERSION]

    # 风险 A：temporary 但没有期限 —— 会被当成永不过期
    risk_a = [(i, md, d) for i, md, d in legacy if md.get("durability") == "temporary"]
    # 风险 B：metadata 里没有 subject —— 一律按"关于用户本人"处理
    risk_b = [(i, md, d) for i, md, d in legacy if "subject" not in md]

    print(f"collection : {store.collection_name}")
    print(f"总条数     : {total}")
    print(f"已迁移     : {len(migrated)}  (schema_version={SCHEMA_VERSION})")
    print(f"待检查     : {len(legacy)}  (没有 schema_version)")

    print(f"\n[风险 A] temporary 但无 valid_until：{len(risk_a)} 条 —— 这些会永不过期")
    for i, md, d in risk_a:
        print(f"  {i}\n    {d}\n    durability=temporary created_at={md.get('created_at')}")

    print(f"\n[风险 B] 缺 subject：{len(risk_b)} 条 —— 一律被当作关于用户本人")
    print("  下面逐条列出，请人工判断内容里说的是不是别人（是→subject=other + subject_name）")
    for i, md, d in risk_b:
        print(f"  {i}\n    {d}")

    need = len(risk_a)
    if need == 0 and not risk_b:
        print("\n结论：没有需要迁移的数据。老数据的默认值恰好就是它们的正解。")
        return 0
    if need == 0:
        print(
            "\n结论：没有【必须修】的数据。"
            "\n      风险 B 那几条只需人工扫一遍：如果内容确实说的都是用户本人，"
            "\n      默认值 subject=user 就是对的，不必改库（改了也只是把隐式默认写成显式）。"
        )
        return 0
    print(f"\n结论：{len(risk_a)} 条【必须修】（风险 A：临时记忆会永不过期）。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
