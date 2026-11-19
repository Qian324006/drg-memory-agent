"""手工试跑：只跑「提取长期记忆 -> 打分 -> 写入记忆库」这一段（不启动对话）。

运行方式（在项目根目录 drg_memory_agent 下）：
    python -m scripts.try_memory_write
"""

from dotenv import load_dotenv

from app.config import ENV_FILE

load_dotenv(ENV_FILE, override=True)
import os
from app.agents.chat_agent import ChatAgent
from langchain_deepseek import ChatDeepSeek
from app.memory.store import MemoryStore
from app.memory.models import MemoryDecision,MemoryItem,EXTRACT_PROMPT,compute_score
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

THRESHOLD =0.6

llm_deepseek = ChatDeepSeek(
        model='deepseek-flash',
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
jude_llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0,
    extra_body={
        "thinking": {"type": "disabled"}
    }
)

user_input = '你好啊，我现在在长沙'
store = MemoryStore()
extractor = jude_llm.with_structured_output(MemoryDecision)
decision = extractor.invoke([SystemMessage(content=EXTRACT_PROMPT),
                                     HumanMessage(content=f"对话内容：\n用户：{user_input}"),])
if not decision.store or not decision.memories:
    print("LLM 判断：无需长期存储")

    # 逐条算分 + 决定
print("开始最终决策")

source_turn = datetime.now().strftime("%Y%m%d_%H%M%S")
memories_to_store = []
agent = ChatAgent(model=llm_deepseek,tools=[])
for m in decision.memories:
    score = compute_score(m)
    if score >= THRESHOLD:
        memories_to_store.append({
            'content': m.content,
            'type': m.type,
            'key': m.key,
            'durability': m.durability,
            'score': score,
        })
if  memories_to_store:
    store.add_many(user_id=agent.thread_id, memories=memories_to_store, source_turn=source_turn)
    print("已存入")

print("总记录数:", store.vector_store._collection.count())
