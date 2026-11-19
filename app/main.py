import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from app.agents.chat_agent import ChatAgent
from app.config import ENV_FILE, LOG_DIR
from app.memory.models import MemoryDecision, build_extract_prompt, compute_score
from app.memory.store import MemoryStore
from app.tools.builtin_tools import search_drg_knowledge

THRESHOLD = 0.6

# 记忆准入：过了闸门还不够，分数要过阈值才写库。
# 0.6 来自 eval/calibrate_weights.py 的校准结果（96 条人工标注，先验权重未被推翻）。

# ---------------------------------------------------------------------------
# 日志
#
# 两个必须踩住的坑：
#   1. filename 必须是真实路径。写成 filename='LOG_DIR'（字符串字面量）会在
#      当前目录生成一个名叫 LOG_DIR 的文件，真正的 logs/ 目录永远是空的。
#   2. filename 指向的目录必须已经存在，否则 logging 打开文件直接抛异常。
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    filename=str(LOG_DIR / "app.log"),
    filemode="a",
    encoding="utf-8",
)

# 第三方库默认也是 INFO，会把每一次 HTTP 请求都写进日志，
# 几千行里找不到自己那一行 ERROR，等于白记。
for _noisy in ("httpx", "httpcore", "openai", "chromadb", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

load_dotenv(ENV_FILE, override=True)

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


def build_llms() -> tuple[ChatDeepSeek, ChatDeepSeek]:
    """构造两个客户端：一个聊天，一个抽取记忆。

    这里不包 try/except —— 构造客户端不发任何网络请求，只做参数校验。
    真正的网络失败发生在 invoke 和 embedding 调用处，那里各自有兜底。
    在这里假装能拦住"网络问题"，只会在真出网络问题时把人带偏。
    """
    if not DEEPSEEK_BASE_URL or not DEEPSEEK_API_KEY:
        print("没有读到 DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY，请检查 .env 配置")
        raise SystemExit(1)

    chat_llm = ChatDeepSeek(
        model="deepseek-flash",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        max_retries=3,
    )

    extract_llm = ChatDeepSeek(
        model="deepseek-v4-flash",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0,
        max_retries=3,
        extra_body={"thinking": {"type": "disabled"}},
    )

    return chat_llm, extract_llm


if __name__ == "__main__":
    chat_llm, extract_llm = build_llms()

    agent = ChatAgent(model=chat_llm, tools=[search_drg_knowledge])
    store = MemoryStore()
    extractor = extract_llm.with_structured_output(MemoryDecision)

    print("你好！有什么可以帮助你的")

    try:
        while True:
            print("\n请输入：")

            try:
                user_input = input().strip()
            except EOFError:
                # 输入流被关闭（管道/重定向输入结束），和空回车一样正常退出
                break

            if not user_input:
                break

            # 记忆是增强项，不是必需项：取不到就降级成"无记忆回答"。
            # 这里不能 continue —— 那等于让向量库的一次抖动变成"用户问不了话"。
            try:
                long_memories = store.search(user_id=agent.thread_id, query=user_input)
                logger.info(f"召回记忆 {len(long_memories)} 条")
            except Exception as e:
                logger.error(f"加载长期记忆失败，本轮按无记忆继续：{e}")
                long_memories = []

            # 记忆块走 state 字段（每轮覆盖），不走 messages（追加）。
            # 塞进 messages 会被 add_messages 永久写进 checkpoint，越聊越长。
            config = {"configurable": {"thread_id": agent.thread_id}}

            try:
                result = agent.chat_agent.invoke(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "memory_block": agent.build_system_prompt(long_memories),
                    },
                    config=config,
                )
            except Exception as e:
                logger.error(f"对话调用失败：{e}")
                print("刚刚好像走神了，没听清。你再说一次？")
                continue

            print(result["messages"][-1].content)

            try:
                decision = extractor.invoke(
                    [
                        SystemMessage(content=build_extract_prompt()),
                        HumanMessage(content=f"对话内容：\n用户：{user_input}"),
                    ]
                )
            except Exception as e:
                logger.error(f"记忆提取失败，本轮不写入：{e}")
                print("\n（这次聊的内容没能整理成记忆，不影响上面这次回答）")
                continue

            if not decision.store or not decision.memories:
                continue

            source_turn = datetime.now().strftime("%Y%m%d_%H%M%S")
            memories_to_store = []

            for m in decision.memories:
                score = compute_score(m)
                if score >= THRESHOLD:
                    memories_to_store.append(
                        {
                            "content": m.content,
                            "type": m.type,
                            "key": m.key,
                            "durability": m.durability,
                            "score": score,
                            "subject": m.subject,
                            "subject_name": m.subject_name,
                            "valid_until": m.valid_until,
                        }
                    )

            if memories_to_store:
                try:
                    store.add_many(
                        user_id=agent.thread_id,
                        memories=memories_to_store,
                        source_turn=source_turn,
                    )
                    logger.info(
                        f"写入 {len(memories_to_store)} 条记忆："
                        f"{[(m['key'], m['subject'], m['valid_until']) for m in memories_to_store]}"
                    )
                except Exception as e:
                    logger.error(f"写入长期记忆失败：{e}")
                    print("\n（这条记忆没能存下来，不影响本次回答）")

    except KeyboardInterrupt:
        print("\n👋 再见！")
