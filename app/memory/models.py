from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from datetime import date, timedelta
import re

class MemoryItem(BaseModel):
    content: str = Field(description="第三人称简洁事实，如 '用户偏好本帮菜'")
    type: Literal["semantic", "procedural", "episodic"] = Field(
        description="semantic=用户事实/偏好; procedural=工具执行模式; episodic=任务经验"
    )
    durability: Literal["permanent", "long_term", "temporary", "one_off"] = Field(
        description="permanent=永久; long_term=长期可变; temporary=有期限; one_off=一次性"
    )
    key: Optional[str] = Field(
        default=None,
        description="同类事实的键，如 city、cuisine_pref；无明确归类则留空"
    )

    # ---------------------------------------------------------------------
    # v2 新增两个维度。它们不参与打分，只决定「什么时候可见、对谁可见」。
    #
    # 为什么不能塞进打分：96 条标注已经证明，权重和阈值都不可分（阈值 0.60→0.92
    # 在留出集上一个误报都没少，只少了 2 个真阳性）。因为"该不该存"和"以什么方式
    # 存在"是两个问题，用同一把尺子量，必然要么漏要么污染。
    # 所以：score 管准入，metadata 管可见性。
    # ---------------------------------------------------------------------
    subject: Literal["user", "other"] = Field(
        default="user",
        description="事实主体：user=用户本人；other=用户之外的具体他人。"
                    "'我妹妹今年高考' 应为 other，否则会混进用户画像、被当成用户自己的事"
    )
    subject_name: Optional[str] = Field(
        default=None,
        description="subject=other 时必填，用关系或称呼表示，如 妹妹、妈妈、张工。"
                    "泛指（朋友/同事/别人）没有检索价值，不要存"
    )
    valid_until: Optional[str] = Field(
        default=None,
        description="事实有效期截止日，YYYY-MM-DD。有明确期限的必须填，"
                    "如'下个月面试'按今天推算到月末。过期后不再被检索，但不是删除"
    )

    explicitness: Literal["explicit", "inferred"] = Field(
        description="explicit=用户明说; inferred=推断"
    )
    user_specific: bool = Field(description="是否关于该用户")
    action_relevance: Literal["high", "medium", "low"] = Field(
        description="是否影响未来行为决策"
    )

class MemoryDecision(BaseModel):
    store: bool = Field(description="是否值得长期保存")
    memories: List[MemoryItem] = Field(default_factory=list)



EXTRACT_PROMPT = """你是一个长期记忆提取器。分析用户消息，判断哪些信息值得写入长期记忆。
值得存：稳定事实、偏好、长期目标。
不要存：寒暄、一次性请求、临时状态、敏感信息。
content 用第三人称简洁表述。不要输出分数，只输出分类标签。
如果没有值得存的内容，store=false，memories=[]。
"""

# 生产用的提取 prompt。和上面那个的区别只有一个：注入了「今天」，否则
# "下个月" 无法换算成 valid_until。
#
# 为什么保留两份而不是把 EXTRACT_PROMPT 改成模板：评测和标注工具必须用**冻结**的
# prompt，否则今天跑出的 96 条分数，明天因为日期变了就复现不了。可复现性优先。
EXTRACT_PROMPT_V2 = """你是一个长期记忆提取器。今天是 {today}。分析用户消息，判断哪些信息值得写入记忆。

判定顺序（先判主体，再判是否该存）：
1. 关于用户本人的稳定事实 / 偏好 / 长期目标 → subject="user"
2. 关于用户之外**具体的**某个人 → subject="other"，subject_name 填关系或称呼（妹妹、妈妈、张工）。
   泛指的人（朋友、同事、别人）没有检索价值，直接不存。
3. 有明确时间点或期限的事（下个月面试、这周出差）→ durability="temporary"，
   valid_until 填截止日 YYYY-MM-DD，从 {today} 往后推算。
4. 寒暄、一次性请求、敏感信息（证件号、手机号、密码）→ 不存。

content 用第三人称简洁表述，不要输出分数，只输出分类标签。
没有值得存的内容就 store=false，memories=[]。
"""


def build_extract_prompt(today: Optional[date] = None) -> str:
    """注入当天日期后的提取 prompt。app/main.py 用这个，评测用冻结的 EXTRACT_PROMPT。"""
    today = today or date.today()
    return EXTRACT_PROMPT_V2.format(today=today.isoformat())


# temporary 类事实没有显式期限时给一个保底寿命。
# 不设的话「我最近在准备面试」会永远留在库里，三个月后再被检索出来就是幻觉来源。
DEFAULT_TTL_DAYS = 90

# 泛指的主体：说了等于没说，还会污染检索。
_GENERIC_SUBJECTS = {"朋友", "同事", "同学", "别人", "有人", "他", "她", "他们", "对方"}

# =============================================================================
# 记忆准入打分
#
# 这里要回答的问题不是「权重填多少」，而是「什么算值得长期记住」。
# 把它拆成两件事，顺序不能反：
#
#   1) 闸门（硬规则）：一票否决项。不满足就直接 0 分，不参与加权。
#      —— 不是关于用户本人的事实、一次性请求、敏感信息。
#   2) 加权求和：在「都该存」的样本里排出轻重。
#      —— 决定多条记忆同时命中时，谁先被塞进 system prompt。
#
# 把否决项混进加权项会出问题：寒暄「你好呀」也能拿到 0.29 分，
# 离 0.6 的阈值只差一点点，稍微改个权重就漏进去。
#
# 权重当前是「专家先验」，每个数值都在下面写了依据。它不是拍脑袋，
# 但也不是从数据里学出来的 —— 想在面试里说「0.4 是有依据的」，依据在这里；
# 想说「0.4 是最优的」，需要 60 条以上标注 + eval/calibrate_weights.py 的校准结果。
# =============================================================================

# durability 权重最高：它决定这条记忆能活多久，是四维里唯一带"时间"属性的。
WEIGHT_DURABILITY = 0.40
# explicitness 次之：用户明说的可以直接用，推断出来的可能要错，错了会污染后续所有回答。
WEIGHT_EXPLICITNESS = 0.25
# action_relevance：是否影响未来行为决策。聊十年也不影响决策的，优先级自然低。
WEIGHT_ACTION_RELEVANCE = 0.20
# user_specific 给最低权重：它是"门槛"不是"程度"，用加权表示本身就偏弱，
# 所以真正的一票否决放在 passes_gate 里做，这里只保留一点区分度。
WEIGHT_USER_SPECIFIC = 0.15

DURABILITY_SCORE = {"permanent": 1.0, "long_term": 0.8, "temporary": 0.4, "one_off": 0.0}
EXPLICITNESS_SCORE = {"explicit": 1.0, "inferred": 0.5}
ACTION_SCORE = {"high": 1.0, "medium": 0.6, "low": 0.2}

# 敏感信息：这类内容不该进长期记忆，和"重要度"无关 —— 再重要也不能存。
# 阈值救不了它（分数会是满分 1.0），所以必须在打分前拦掉。
#
# 2026-11-19 补了两类：工号 / 学号、家庭详细地址。
# 起因是评测里翻出两条满分误存：「我工号是 A20230149」「我家地址是长沙市岳麓区某小区 3 栋 502」。
# 它们不是"重要度判断错了"，是这两类信息当时根本没被拦。
#
# 地址那条为什么写成 `\d+\s*(栋|号楼|单元|室|层)` 而不是直接匹配"室"：
# 直接匹配"室"会把「我室友学的是临床医学」也拦掉（"室友"里有个"室"）。
# 要求前面必须有数字，才能区分门牌号和"室友"。
_SENSITIVE_PATTERNS = [
    re.compile(r"\d{17}[\dXx]"),        # 身份证 18 位
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),  # 手机号 11 位
    re.compile(r"\d{16,19}"),           # 银行卡
    re.compile(r"(密码|口令|验证码|银行卡号)"),  # 凭据类关键词
    re.compile(r"(工号|学号|员工编号|员工号|学籍号|工牌号)"),  # 工号 / 学号
    re.compile(r"(家庭住址|详细地址|家庭地址|居住地址|门牌号|收件地址)"),  # 地址关键词
    re.compile(r"\d+\s*(栋|号楼|单元|室|层)"),  # 门牌号：3 栋 502 / 12 号楼 3 单元
]


def is_sensitive(text: str) -> bool:
    """命中敏感信息返回 True。"""
    return any(p.search(text or "") for p in _SENSITIVE_PATTERNS)


def is_expired(valid_until: Optional[str], today: Optional[date] = None) -> bool:
    """有效期是否已过。

    解析失败时返回 False（当作没过期）：宁可多留一条，也不要因为格式问题
    静默丢掉用户数据。过期只在检索时生效，不删除 —— 保留下来才能回答
    "它是不是该过期"。
    """
    if not valid_until:
        return False
    today = today or date.today()
    try:
        deadline = date.fromisoformat(valid_until.strip()[:10])
    except ValueError:
        return False
    return deadline < today


def passes_gate(m: MemoryItem) -> bool:
    """闸门：通过才有资格进入加权打分。

    三个例外值得注意：
    - 不关于本人的事实（semantic/episodic）不该存进"用户记忆"，但 procedural
      例外：用户命令式的偏好（如"回答先给结论"）常被抽取成不关于本人的表述，
      按 user_specific 一刀切会漏掉真实偏好。
    - subject="other" 是 v2 加的第二条例外："我妹妹今年高考" 对用户有用，但不是
      用户本人的事实。要求必须带具体称呼 —— 泛指主体既检索不到，又容易被 LLM
      当作用户自己的事，比不存更糟。
    - 敏感信息无论分数多高都要拦。
    """
    if m.durability == "one_off":
        return False
    if m.subject == "other":
        name = (m.subject_name or "").strip()
        if not name or name in _GENERIC_SUBJECTS:
            return False
    elif not m.user_specific and m.type != "procedural":
        return False
    if is_sensitive(m.content):
        return False
    return True


def weighted_score(m: MemoryItem) -> float:
    """只做加权求和，不含闸门。留着单独可调用，方便校准脚本和调试。"""
    s = 0.0
    s += WEIGHT_DURABILITY * DURABILITY_SCORE[m.durability]
    s += WEIGHT_EXPLICITNESS * EXPLICITNESS_SCORE[m.explicitness]
    s += WEIGHT_ACTION_RELEVANCE * ACTION_SCORE[m.action_relevance]
    s += WEIGHT_USER_SPECIFIC * (1.0 if m.user_specific else 0.0)
    return round(s, 3)


def compute_score(m: MemoryItem) -> float:
    """记忆准入分：0 = 不该存，越接近 1 = 越值得存。

    调用方（app/main.py）只判断 `score >= THRESHOLD`，所以闸门必须在这里生效：
    没过闸门的记忆直接 0 分，永远低于任何合理阈值。
    """
    if not passes_gate(m):
        return 0.0
    return weighted_score(m)
