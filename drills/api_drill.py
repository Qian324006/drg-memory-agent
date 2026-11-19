"""练习文件：从零写一个最小的 FastAPI + agent 接口。

这不是项目代码，是练手用的 —— 比 app/api.py 少了一切：没有 sources、
没有后台抽取、没有加锁、没有多会话。只有一条最细的主线：

    启动时造模型和 agent  →  请求进来 invoke  →  取最后一条消息返回

第一版有 5 个 bug，全在「层与层之间」。每个都用【bug N】标出来了。
下一遍请闭卷重写这个文件，写不出来再回来看。
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

# 【bug 2 的修法】原来这行是：
#     from app.agents.chat_agent import ChatAgent, build_system_prompt
# build_system_prompt 是 ChatAgent 的**方法**，不是模块级函数，import 当场炸：
#     ImportError: cannot import name 'build_system_prompt' from 'app.agents.chat_agent'
# 方法只能挂在对象上调用：agent.build_system_prompt(...)
#
# （另外，bug 1 不在这个文件里：原来这个文件放在 model/fastapi_practice/ 下面，
#   从那个目录启动时 `from app...` 找不到 app —— 报 ModuleNotFoundError: No module named 'app'。
#   跨目录用项目里的包，必须站在项目根目录（drg_memory_agent）下启动。）
from app.agents.chat_agent import ChatAgent
from app.config import ENV_FILE
from app.memory.store import MemoryStore
from app.tools.builtin_tools import search_drg_knowledge

# 【bug 3 的修法】main.py 第 45 行就有这句 load_dotenv，但那是 main.py 的顶层代码，
# 你自己写的新文件里没有它 —— 于是下面两个 getenv 拿到 None。
# 这个 bug 最阴：代码看着完全正常，报错却发生在 ChatDeepSeek 构造的时候，
# 指向的是「模型配置错了」，而不是「你没读 .env」。
load_dotenv(ENV_FILE, override=True)

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 练习用独立会话，别写进正式聊天历史。
# 注意：ChatAgent 自己身上那个 thread_id 只是个默认字符串，真正决定「算哪个会话」的
# 是 invoke 时 config 里的 thread_id —— api.py 用的是请求体里的，同理。
THREAD_ID = "drill_user"


class Response(BaseModel):
    answer: str = Field(description="回答的内容")


class Request(BaseModel):
    message: str = Field(description="用户说的话")


@asynccontextmanager
async def lifespan(app: FastAPI):
    chat_llm = ChatDeepSeek(
        model="deepseek-flash",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        max_retries=3,
    )

    # 【bug 4 的修法】ChatAgent.__init__(self, model, tools, system_prompt=...)
    # tools 没有默认值，漏了就是：
    #     TypeError: ChatAgent.__init__() missing 1 required positional argument: 'tools'
    #
    # 【bug 7：我一开始让你写 tools=[]，这是个错误的建议】
    # 空列表能让程序跑起来，但答案是垃圾 —— 因为 ChatAgent 默认带着 DRG 系统提示词，
    # 里面写着「专业问题必须检索」，可你一个工具都没给它。模型于是把要调用的工具
    # 当成正文吐了出来，真实输出长这样：
    #     \n\n<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="search_drg_knowledge">...
    # 教训：程序「跑通」不等于「对」。配置之间互相矛盾时，不会报错，只会给错答案。
    app.state.chat_agent = ChatAgent(model=chat_llm, tools=[search_drg_knowledge])

    # 【bug 5 的修法】MemoryStore() 原来写在 chat() 里，等于每个请求重建一次
    # embedding 模型 + 打开一次 Chroma。这和 api.py 注释里那个坑是同一类错误，
    # 只不过踩的是 Chroma 不是 sqlite（所以它不会锁死，只会变慢）。
    # 原则：贵的东西在 lifespan 里建一次，请求里只使用。
    app.state.store = MemoryStore()

    yield

    app.state.chat_agent.conn.close()


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/chat")
def chat(payload: Request) -> Response:
    agent = app.state.chat_agent

    long_memories = app.state.store.search(user_id=THREAD_ID, query=payload.message)

    config = {"configurable": {"thread_id": THREAD_ID}}

    # 【bug 6 的修法】原来写的是 agent.invoke(...)，会报：
    #     AttributeError: 'ChatAgent' object has no attribute 'invoke'
    # 这里是两层，你混成了一层：
    #   ChatAgent  ← 你自己写的类，负责「建和管」：连 sqlite、拼提示词
    #   chat_agent ← create_agent(...) 造出来的 LangGraph 图，负责「跑」：invoke / stream
    # 能 invoke 的是 agent.chat_agent，不是 agent。
    result = agent.chat_agent.invoke(
        {
            "messages": [HumanMessage(content=payload.message)],
            "memory_block": agent.build_system_prompt(long_memories),
        },
        config=config,
    )

    return Response(answer=result["messages"][-1].content)
