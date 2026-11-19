from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.knowledge.database import knowledge_db

class Search_schema(BaseModel):
    query:str = Field(description="用户的提问内容，例如 '不稳定病组怎么定义' 或 '医保基金包括哪几类'")

@tool(args_schema=Search_schema,description="当用户询问关于 DRG 医保、病组分类、医保基金等专业医疗政策问题时，调用此工具从官方知识库中检索相关的规定和答案。")
def search_drg_knowledge(query: str) -> str:
    try:
        results = knowledge_db.similarity_search_with_relevance_scores(query, k=3)
    except Exception as err:
        return f"搜索失败，原因{err},告知用户暂时不能回答"
    if not results:
        return f"知识库中未检索到相关条款"
    formatted = []
    for doc, relevance in results:
        level = "高" if relevance >= 0.80 else ("中" if relevance >= 0.70 else "低")
        formatted.append(rf"[证据等级：{level}｜相关性：{relevance:.3f}]\内容:{doc.page_content}")
    return "\n\n".join(formatted)
