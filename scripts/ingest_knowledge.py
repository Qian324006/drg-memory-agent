"""把 data/raw 里的医保问答语料导入 DRG 知识库（Chroma）。

运行方式（在项目根目录 drg_memory_agent 下）：
    python -m scripts.ingest_knowledge
"""

import json

from app.config import DATA_RAW_DIR
from app.knowledge.database import knowledge_db

json_file_path = DATA_RAW_DIR / "medical-insurance-alpaca-cleaned.json"

with open(json_file_path, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

texts = []
metadatas = []
ids = []

for i,item in enumerate(qa_data):
    instruction = item.get("instruction", "").strip()
    output = item.get("output", "").strip()

    # 跳过空数据
    if not instruction or not output:
        continue

    # 拼接用于 embedding 的全文（用换行符隔开，保留语义结构）
    text = f"问题：{instruction}\n回答：{output}"

    texts.append(text)
    metadatas.append({
        "question": instruction,  # 原数据存 metadata，方便展示
        "answer": output,  # 原数据存 metadata
        "type": "domain_knowledge"  # 打个标签，方便以后过滤
    })
    ids.append(f"drg_qa_{i}")  # 唯一的 ID

print(f"✅ 数据解析完成，共准备导入 {len(texts)} 条记录。")

BATCH_SIZE = 10

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i: i + BATCH_SIZE]
    batch_metadatas = metadatas[i: i + BATCH_SIZE]
    batch_ids = ids[i: i + BATCH_SIZE]

    knowledge_db.add_texts(
        texts=batch_texts,
        metadatas=batch_metadatas,
        ids=batch_ids
    )
    print(f"⏳ 已导入: {min(i + BATCH_SIZE, len(texts))} / {len(texts)} 条...")

# ================= 6. 验证结果 =================
print("🎉 知识库导入完成！")
print(f"当前知识库总记录数: {knowledge_db._collection.count()}")
