from datetime import datetime, date, timedelta
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import Optional
import uuid
from dotenv import load_dotenv
import os
import logging
from app.config import ENV_FILE, MEMORY_DB_DIR
from app.memory.models import DEFAULT_TTL_DAYS, is_expired

# 写入时打上的 schema 版本。v1 的老记忆没有这个字段 —— 它就是"没迁过"的标记，
# 以后要批量修数据，一条查询就能把老数据和迁过的数据分开。
SCHEMA_VERSION = 2
logger = logging.getLogger(__name__)


def _resolve_valid_until(durability: str, valid_until: Optional[str]) -> Optional[str]:
    """算出这条记忆的有效期。

    temporary 类没给期限时，发一个保底寿命（DEFAULT_TTL_DAYS）。
    永久/长期类不设期限，返回 None 表示永不过期。
    """
    if durability != "temporary":
        return valid_until
    if valid_until:
        return valid_until
    return (date.today() + timedelta(days=DEFAULT_TTL_DAYS)).isoformat()


class MemoryStore():
    def __init__(self,model :str = "text-embedding-v3",persist_directory : str = str(MEMORY_DB_DIR),
                 collection_name : str = "demo"):
        self.model = model
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        load_dotenv(ENV_FILE, override=True)
        self.base_url = os.getenv("PROXY_BASE_URL")
        self.api_key = os.getenv("PROXY_API_KEY")


        self.embeddings = OpenAIEmbeddings(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            check_embedding_ctx_length = False,
            max_retries=3
        )

        self.vector_store = Chroma(
        collection_name=self.collection_name,
        embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )

#添加函数

    def add(self,user_id: str,content: str,type: str = "semantic",key: Optional[str] = None,
            durability: str = "long_term",score: float = 0.0,source_turn: Optional[str] = None,
            memory_id: Optional[str] = None,
            subject: str = "user", subject_name: Optional[str] = None,
            valid_until: Optional[str] = None,
    ) -> str:
        """
                存入一条记忆。

                参数：
                  user_id      必填，记忆归属的用户
                  content      必填，第三人称事实描述，如 "用户偏好本帮菜"
                  type         semantic / procedural / episodic
                  key          同类事实的键，如 "city"、"cuisine_pref"
                  durability   permanent / long_term / temporary
                  score        提取器算出的重要度（0-1）
                  source_turn  来源对话 ID，便于回溯
                  memory_id    可选，不传则自动生成
                  subject      user / other，事实主体
                  subject_name subject=other 时必填，如 "妹妹"
                  valid_until  有效期截止日 YYYY-MM-DD，过期后不再被检索

                返回：写入的 memory_id
                """

        if not user_id or not content.strip():
            raise ValueError("user_id 和 content 不能为空")

            # 生成唯一 ID：user_id + key 保证同 key 覆盖
        if memory_id is None:
            key_part = key or uuid.uuid4().hex[:8]
            memory_id = f"mem_{user_id}_{key_part}"

        metadata = {
            "user_id": user_id,
            "type": type,
            "key": key,
            "durability": durability,
            "score": score,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "source_turn": source_turn,
            "subject": subject,
            "subject_name": subject_name,
            "valid_until": _resolve_valid_until(durability, valid_until),
            "schema_version": SCHEMA_VERSION,
        }
        try:
            self.vector_store._collection.upsert(
                documents=[content],
                metadatas=[metadata],
                ids=[memory_id],
                embeddings=self.embeddings.embed_documents([content])
            )
        except Exception as e:
            logger.error(f"写入失败{e}")
        return memory_id

    def add_many(self, user_id: str, memories: list[dict],source_turn :Optional[str] = None) -> list[str]:
        """
        批量写入。memories 是 dict 列表，字段同 add() 的参数。
        适合从 LLM 提取器拿到 MemoryDecision 后一次性写入。
        """
        if not memories:
            return []

        texts, metadatas, ids = [], [], []
        for m in memories:
            key_part = m.get("key") or uuid.uuid4().hex[:8]
            mid = m.get("memory_id") or f"mem_{user_id}_{key_part}"
            durability = m.get("durability", "long_term")
            texts.append(m["content"])
            metadatas.append({
                "user_id": user_id,
                "type": m.get("type", "semantic"),
                "key": m.get("key"),
                "durability": durability,
                "score": m.get("score", 0.0),
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "source_turn": m.get("source_turn"),
                "subject": m.get("subject", "user"),
                "subject_name": m.get("subject_name"),
                "valid_until": _resolve_valid_until(durability, m.get("valid_until")),
                "schema_version": SCHEMA_VERSION,
            })
            ids.append(mid)

        try:
            self.vector_store._collection.upsert(documents=texts,
                                                 metadatas=metadatas,
                                                 ids=ids,
                                                 embeddings=self.embeddings.embed_documents(texts))
        except Exception as e:
            logger.error(f"写入失败{e}")
            raise e
        return ids

    def search(self,user_id: str,query:str,k:int = 5, include_others: Optional[bool] = None,
               today: Optional[date] = None )-> list[dict]:
        """检索这条用户当前可用的记忆。

        两层过滤（都发生在召回之后，不是写入时删掉）：
          1. 过期：valid_until 已过的不再返回 —— 「下个月的面试」下个月之后就不该
             再影响回答，但记录要留着，用于回答「当时为什么那么建议」。
          2. 他人：subject=other 的记忆默认不返回，除非这句话本身提到了那个人
             （「我妹妹」）。这是防幻觉的关键：不这么滤，问「我最近该做什么」时
             「妹妹今年高考」会被当成用户自己的事讲出来。

        为什么多取 4 倍再在 Python 侧过滤：Chroma 对「元数据里没有该字段」的老数据
        用 $eq 匹配不上，把 subject 直接写进 filter 会把加字段之前写的记忆全部
        静默隐藏。多取一点换正确性，记忆库规模上用户级过滤够用。
        """
        fetch_k = max(k * 4, k)
        results = []
        try:
            results = self.vector_store.similarity_search_with_score(query=query,
                                                                    k=fetch_k,
                                                                    filter={"$and": [{"user_id": user_id},{"status": "active"}]}
                                                                    )
        except Exception as e:
            logger.error(f"长期记忆加载失败，可能是embedding模型挂了，错误为：{e}")
        picked = []
        for doc, score in results:
            md = doc.metadata or {}
            if is_expired(md.get("valid_until"), today):
                continue
            if md.get("subject", "user") != "user":
                name = md.get("subject_name") or ""
                if not (include_others or (name and name in query)):
                    continue
            picked.append({"content":doc.page_content,"score":score,"metadata":md})
            if len(picked) >= k:
                break
        return picked
