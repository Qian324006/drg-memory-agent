from datetime import date
import sqlite3

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, ModelRequest, dynamic_prompt
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.system_prompt import DRG_SYSTEM_PROMPT
from app.config import DATA_DIR


class AgentStateWithMemory(AgentState):
    """每轮被覆盖的记忆块。

    memory_block 是 LastValue（覆盖），不是 add_messages（追加），所以它不会越聊越长；
    它也不进 messages，因此不会被写进 checkpoint。
    """
    memory_block: str


class ChatAgent():
    # agent创建的初始化
    def __init__(self, model : ChatDeepSeek , tools : list , system_prompt : str = DRG_SYSTEM_PROMPT):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt

        # 会话历史落盘：进程重启后同一个 thread_id 还能接着聊
        db_path = DATA_DIR / "checkpoints.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.memory = SqliteSaver(self.conn)

        base = self.system_prompt

        @dynamic_prompt
        def with_memory(request: ModelRequest) -> str:
            # 只在这一次模型调用时拼上记忆块，不写进 messages，
            # 所以 checkpoint 里永远只有 Human / AI 两种消息。
            return base + request.state.get("memory_block", "")

        self.chat_agent = create_agent(
            model = self.model,
            tools = self.tools,
            system_prompt = self.system_prompt,
            state_schema = AgentStateWithMemory,
            middleware = [with_memory],
            checkpointer = self.memory,
        )
        self.thread_id = "default_user"

    def build_system_prompt(self, memories: list[dict]) -> str:
        """把本轮召回到的记忆渲染成一段附加提示块（不是完整的系统提示词）。"""
        today_str = date.today().isoformat()

        lines = []
        for m in memories:
            content = m.get("content", "")
            metadata = m.get("metadata", {})
            if metadata.get("subject", "user") == "user":
                prefix = ""
            else:
                prefix = f"（关于{metadata.get('subject_name') or '他人'}，不是用户本人的事）"

            valid_until = metadata.get("valid_until")
            time_tag = ""
            if valid_until:
                try:
                    days = (date.fromisoformat(str(valid_until)) - date.today()).days
                    time_tag = f" [有效期至 {valid_until}，还剩 {days} 天]"
                except ValueError:
                    time_tag = f" [有效期至 {valid_until}]"

            lines.append(f"- {prefix}{content}{time_tag}")

        # 前面的 \n\n 是为了和基础系统提示词分开：这段是拼在它后面的
        header = f"\n\n【当前系统时间】今天是 {today_str}。判断记忆是否临近过期时以此为准。"
        if not lines:
            return header
        return (
                header
                + "\n【关于该用户的长期记忆】标了「关于某人」的不是用户本人的事：\n"
                + "\n".join(lines)
        )
