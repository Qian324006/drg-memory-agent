"""把 data/checkpoints.sqlite 里的内容读成人话。

checkpoints.sqlite 里存的是 msgpack 二进制快照，用文本编辑器或 DB 浏览器打开
只能看到乱码 —— 它不是坏文件，而是 LangGraph 把整个 state 序列化后存下的。
这个脚本用 SqliteSaver 把它读回来。

用法（在 drg_memory_agent 目录下）：
    python -m scripts.read_checkpoints
    python -m scripts.read_checkpoints --thread default_user --limit 4
    python -m scripts.read_checkpoints --full          # 不截断长回答
"""
import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from app.config import DATA_DIR  # noqa: E402

DB_PATH = DATA_DIR / "checkpoints.sqlite"


def main() -> None:
    parser = argparse.ArgumentParser(description="读取会话历史")
    parser.add_argument("--thread", help="只看某个 thread_id，默认看全部")
    parser.add_argument("--limit", type=int, default=10, help="每个会话显示最近 N 条消息，默认 10")
    parser.add_argument("--full", action="store_true", help="不截断消息内容")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"还没有会话记录：{DB_PATH} 不存在（聊过一轮之后才会有）")
        return

    print(f"文件: {DB_PATH}")
    print(f"大小: {DB_PATH.stat().st_size} 字节")

    raw = sqlite3.connect(str(DB_PATH))
    threads = [r[0] for r in raw.execute(
        "select distinct thread_id from checkpoints order by thread_id"
    )]
    print(f"会话数: {len(threads)}  ->  {threads}\n")

    saver = SqliteSaver(sqlite3.connect(str(DB_PATH), check_same_thread=False))

    for thread_id in threads:
        if args.thread and thread_id != args.thread:
            continue
        tup = saver.get_tuple({"configurable": {"thread_id": thread_id}})
        if tup is None:
            print(f"[{thread_id}] 读不到状态")
            continue

        values = tup.checkpoint.get("channel_values", {})
        messages = values.get("messages", [])
        shown = messages[-args.limit:]

        print("=" * 72)
        print(f"[{thread_id}]  共 {len(messages)} 条消息，显示最近 {len(shown)} 条")
        for i, m in enumerate(shown, start=len(messages) - len(shown) + 1):
            content = str(m.content)
            if not args.full and len(content) > 160:
                content = content[:160] + " …"
            print(f"  {i:>2}. {type(m).__name__:<14} {content}")
            for call in getattr(m, "tool_calls", None) or []:
                print(f"      -> 调用工具 {call.get('name')}({call.get('args')})")

        block = values.get("memory_block")
        if block:
            print("  本轮注入的记忆块:")
            for line in block.strip().splitlines():
                print("    " + line)

        dumped = sum(1 for m in messages if type(m).__name__ == "SystemMessage")
        print(f"  体检: SystemMessage {dumped} 条（应为 0，系统提示词不该进历史）")

    raw.close()


if __name__ == "__main__":
    main()
