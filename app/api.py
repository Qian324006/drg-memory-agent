"""HTTP 接口层：把 app/main.py 里那个 while 循环，原样搬到 Web 上。

main.py 回答的问题是「在终端里怎么一轮一轮聊下去」。
api.py 回答的问题是「别的程序怎么问同一套问题」——网页、小程序、同事的脚本。

搬过来只有三件事必须做对，做错任何一件上线就是坑：

  1. 重活儿只在进程启动时干一次。
     ChatAgent 每 new 一次就新开一个 sqlite 连接。写在请求里，就是每个请求
     开一个连接，并发一上来立刻 'database is locked'。
     → 放进 lifespan，全进程只有一个。

  2. 别用 async def 装同步调用。
     agent.invoke 内部是同步阻塞的（底层是同步 HTTP 客户端，调用即等待）。
     写成 async def 不会让它变异步，只会让它卡在事件循环上——整个进程唯一的
     调度员被占住，连 /health 都回不了。
     写成普通 def，FastAPI 会自动丢到线程池，事件循环继续服务别人。

  3. 记忆抽取别挡在用户前面。
     抽取要再调一次模型，等它等于把回复时间翻一倍。放进 BackgroundTasks，
     答案先返回，记忆慢慢写。
"""

import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from app.agents.chat_agent import ChatAgent
from app.main import THRESHOLD, build_llms
from app.memory.models import MemoryDecision, build_extract_prompt, compute_score
from app.memory.store import MemoryStore
from app.tools.builtin_tools import search_drg_knowledge

logger = logging.getLogger("app.api")

# 同一个 sqlite 文件被多个线程同时写，会直接报 database is locked。
# 一把锁把对话串起来，代价是同一时刻只能有一个用户在等模型。
# 真实系统该做的是「一个会话一个连接」或按会话分库；这个项目是单人演示，不折腾。
_db_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 请求 / 响应格式
#
# 这几个类不是给 Python 用的，是给 /docs 和调用方看的：
# 声明一次，接口文档、参数校验、错误提示全都自动生成。
# 类型写错一个字（比如 message 写成 int），FastAPI 直接返回 422，不会进函数体。
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="用户这一轮说的话",
        examples=["不稳定病组怎么定义"],
    )
    thread_id: str = Field(
        default="default_user",
        description="会话 id。同一个 id 共享聊天历史和记忆；换一个就是全新会话",
        examples=["default_user"],
    )


class ChatResponse(BaseModel):
    answer: str = Field(description="模型的回答")
    sources: list[str] = Field(
        default_factory=list,
        description="本轮从知识库检索到的原文块（模型看到什么，这里就报什么；没检索就是空数组）",
    )
    memory_hits: int = Field(description="本轮召回了几条长期记忆")
    thread_id: str = Field(description="原样返回，方便调用方确认自己问的是哪个会话")
    elapsed: float = Field(description="本轮耗时（秒），含记忆召回 + 模型生成，不含后台记忆抽取")


class HealthResponse(BaseModel):
    ok: bool
    agent_ready: bool = Field(description="agent、向量库、抽取器是否已经在 lifespan 里建好")


# ---------------------------------------------------------------------------
# 两个小工具
# ---------------------------------------------------------------------------


def _collect_sources(messages: list) -> list[str]:
    """从这一轮的工具调用结果里，摘出知识库命中的原文。

    为什么从 messages 里翻，而不是让工具改成一个返回结构体：
    工具的输出是给模型看的，模型看到什么，sources 就报什么，两边不会走偏。
    要解析成结构化字段就得按格式切字符串，格式一变就悄悄错。

    为什么要切一刀 last_human：
    result["messages"] 不是「这一轮」的消息，而是 checkpoint 里拼出来的全量历史。
    不切的话，上一轮检索到的条款会被算成这一轮的出处。
    """
    last_human = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_human = i

    sources: list[str] = []
    for msg in messages[last_human + 1:]:
        if not isinstance(msg, ToolMessage):
            continue
        for block in str(msg.content).split("\n\n"):
            block = block.strip()
            # 工具失败时会返回「搜索失败…」「未检索到…」，那不是出处，不进 sources
            if block.startswith("[证据等级"):
                sources.append(block)
    return sources


def _recall(store: MemoryStore, thread_id: str, query: str) -> list[dict]:
    """取长期记忆。记忆是增强项，不是必需项 —— 取不到就降级成「无记忆回答」。

    这里绝对不能往上抛：向量库抖一下，用户就问不了话，这是最差的一种失败。
    """
    try:
        memories = store.search(user_id=thread_id, query=query)
        logger.info(f"[{thread_id}] 召回记忆 {len(memories)} 条")
        return memories
    except Exception as err:
        logger.error(f"[{thread_id}] 加载长期记忆失败，本轮按无记忆继续：{err}")
        return []


def _remember(thread_id: str, user_input: str, store: MemoryStore, extractor) -> None:
    """本轮结束后再抽取并写入记忆。

    必须是普通 def：BackgroundTasks 对同步函数会丢进线程池；
    写成 async def 又没有 await，反而占住事件循环（和 /chat 那个坑同一个）。
    """
    try:
        decision = extractor.invoke(
            [
                SystemMessage(content=build_extract_prompt()),
                HumanMessage(content=f"对话内容：\n用户：{user_input}"),
            ]
        )
    except Exception as err:
        logger.error(f"[{thread_id}] 记忆提取失败，本轮不写入：{err}")
        return

    if not decision.store or not decision.memories:
        # 一行日志很重要：没有它，「后台任务跑了但判断不用存」和「后台任务压根没跑」
        # 在日志里长得一模一样，出问题只能靠猜。
        logger.info(f"[{thread_id}] 本轮没有需要记住的内容")
        return

    to_store = []
    for m in decision.memories:
        score = compute_score(m)
        if score >= THRESHOLD:
            to_store.append(
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

    if not to_store:
        return

    try:
        store.add_many(
            user_id=thread_id,
            memories=to_store,
            source_turn=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        logger.info(
            f"[{thread_id}] 写入 {len(to_store)} 条记忆："
            f"{[(m['key'], m['subject'], m['valid_until']) for m in to_store]}"
        )
    except Exception as err:
        logger.error(f"[{thread_id}] 写入长期记忆失败：{err}")


# ---------------------------------------------------------------------------
# 进程生命周期
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时建一次，退出时关一次。

    对照 main.py 第 82-86 行：那里是「启动脚本时」干这些事；
    这里是「启动服务时」干同样的事。搬到 Web 上，一个进程要接待 N 个用户，
    所以只能干一次。
    """
    try:
        chat_llm, extract_llm = build_llms()
    except SystemExit:
        # build_llms 读不到 .env 时 raise SystemExit。
        # 在这里换成一句能看懂的报错，否则启动失败的堆栈里只有一句 "SystemExit: 1"。
        raise RuntimeError("读不到 DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY，检查 .env 后再启动")

    app.state.agent = ChatAgent(model=chat_llm, tools=[search_drg_knowledge])
    app.state.store = MemoryStore()
    app.state.extractor = extract_llm.with_structured_output(MemoryDecision)
    logger.info("lifespan：agent / 向量库 / 抽取器 就绪")

    yield

    app.state.agent.conn.close()
    logger.info("lifespan：服务退出，sqlite 连接已关闭")


app = FastAPI(
    title="DRG 医保知识库 + 长期记忆 Agent",
    description="检索医保政策条文，记住用户说过的事。同一个 thread_id 接着聊。",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["运维"],
    summary="存活探针",
    description="只回答「进程还活着、组件建好没」。不碰向量库，不调模型，所以它可以被高频调用。",
)
def health(request: Request) -> HealthResponse:
    return HealthResponse(ok=True, agent_ready=hasattr(request.app.state, "agent"))


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["会话"],
    summary="问一轮，拿答案",
    description=(
        "同步返回完整答案。记忆抽取在响应发出后才开始，所以某条记忆可能要"
        "下一轮对话才生效——这是用「快」换「立刻能查到」的取舍。"
    ),
)
def chat(payload: ChatRequest, background: BackgroundTasks, request: Request) -> ChatResponse:
    started = time.perf_counter()

    agent: ChatAgent = request.app.state.agent
    store: MemoryStore = request.app.state.store
    extractor = request.app.state.extractor

    memories = _recall(store, payload.thread_id, payload.message)

    # memory_block 是 state 字段（每轮覆盖），不是 messages 里的一条消息。
    # build_system_prompt([]) 也会返回「今天的日期」那一段，所以哪怕没召回记忆，
    # memory_block 每轮也一定被重新赋值 —— checkpoint 里不会残留上一轮的记忆块。
    config = {"configurable": {"thread_id": payload.thread_id}}

    try:
        with _db_lock:
            result = agent.chat_agent.invoke(
                {
                    "messages": [HumanMessage(content=payload.message)],
                    "memory_block": agent.build_system_prompt(memories),
                },
                config=config,
            )
    except Exception as err:
        # 502：错在我背后的上游（模型/网络），不在调用方。
        # 用 500 会让调用方以为是自己的请求写错了。
        logger.error(f"[{payload.thread_id}] 对话调用失败：{err}")
        raise HTTPException(status_code=502, detail="模型调用失败，请重试。这次没有写入任何记忆")

    last = result["messages"][-1]
    answer = last.content if isinstance(last.content, str) else str(last.content)
    sources = _collect_sources(result["messages"])

    # add_task 只是排队：响应结束后 Starlette 才真正执行它，
    # 所以用户看到的延迟里不包含这一次抽取调用。
    background.add_task(_remember, payload.thread_id, payload.message, store, extractor)

    return ChatResponse(
        answer=answer,
        sources=sources,
        memory_hits=len(memories),
        thread_id=payload.thread_id,
        elapsed=round(time.perf_counter() - started, 2),
    )
