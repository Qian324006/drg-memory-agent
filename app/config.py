"""项目路径配置：所有路径只在这里定义一次。

这样其他模块就不用再写 "./chroma_knowledge_db" 这类依赖「当前工作目录」的相对路径，
无论你在哪个目录下启动脚本，路径都是对的。
"""

from pathlib import Path

# app/config.py -> app -> 项目根目录（drg_memory_agent）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据目录：向量库、原始语料都放这里，代码和数据的边界一眼可见
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"
LOGS_DIR =  PROJECT_ROOT / "logs"
LOG_DIR = LOGS_DIR / "log"

# 两个向量库
KNOWLEDGE_DB_DIR = CHROMA_DIR / "knowledge_db"  # 医保 DRG 知识库（由语料导入，只读）
MEMORY_DB_DIR = CHROMA_DIR / "memory_db"        # 用户长期记忆库（运行时会不断写入）

# .env 优先放在项目根目录；找不到就回退到上一级仓库根目录（该文件不在版本库里）
ENV_FILE = PROJECT_ROOT / ".env"
if not ENV_FILE.exists():
    ENV_FILE = PROJECT_ROOT.parents[1] / ".env"
