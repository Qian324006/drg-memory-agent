"""记忆 v2 维度的自检脚本。

跑法（在 drg_memory_agent 目录下）：
    python -m scripts.check_memory_v2

它验证三件事：
  1. 老数据不受影响 —— 没有 subject/valid_until 的 MemoryItem 打分和闸门结果完全不变
  2. 两个真实案例 —— 「妹妹今年高考」进得了闸门，「下个月面试」带上了有效期
  3. 有效期判定 —— is_expired 的边界，以及解析失败时不误删
"""

from datetime import date

from app.memory.models import (
    MemoryItem,
    compute_score,
    is_expired,
    passes_gate,
    weighted_score,
)

OK = "  ✓"
BAD = "  ✗"


def item(**kw) -> MemoryItem:
    base = dict(
        content="用户偏好本帮菜",
        type="semantic",
        durability="long_term",
        key=None,
        explicitness="explicit",
        user_specific=True,
        action_relevance="medium",
    )
    base.update(kw)
    return MemoryItem(**base)


def check(label: str, got, want) -> bool:
    good = got == want
    print(f"{OK if good else BAD} {label}: {got!r}" + ("" if good else f"（期望 {want!r}）"))
    return good


def main() -> int:
    passed = True

    print("\n[1] 老数据回归：不传新字段时必须和 v1 行为一致")
    m = item()
    passed &= check("普通用户事实分数", weighted_score(m), 0.84)
    passed &= check("普通用户事实过闸门", passes_gate(m), True)
    passed &= check("默认 subject", m.subject, "user")
    passed &= check("默认 valid_until", m.valid_until, None)
    passed &= check("不关于用户的事被拦（v1 行为）", passes_gate(item(user_specific=False)), False)
    passed &= check("寒暄（one_off）被拦", passes_gate(item(durability="one_off")), False)
    passed &= check("敏感信息被拦", passes_gate(item(content="用户的手机号 13812345678")), False)

    print("\n[2] 案例一：我妹妹今年高考（关于他人，但有具体称呼）")
    sister = item(
        content="用户的妹妹今年参加高考",
        type="semantic",
        durability="long_term",
        user_specific=False,
        action_relevance="medium",
        subject="other",
        subject_name="妹妹",
    )
    passed &= check("过闸门", passes_gate(sister), True)
    passed &= check("分数达到阈值 0.6", compute_score(sister) >= 0.6, True)
    print(f"    分数 {compute_score(sister)}")

    print("\n[3] 案例二：下个月我准备去面试（有期限）")
    interview = item(
        content="用户准备在 2026 年 11 月参加面试",
        type="episodic",
        durability="temporary",
        action_relevance="high",
        valid_until="2026-11-30",
    )
    passed &= check("过闸门", passes_gate(interview), True)
    passed &= check("分数达到阈值 0.6", compute_score(interview) >= 0.6, True)
    print(f"    分数 {compute_score(interview)}")

    print("\n[4] 泛指的主体被拦")
    vague = item(content="用户的朋友也不吃辣", user_specific=False, subject="other", subject_name="朋友")
    passed &= check("泛指朋友被拦", passes_gate(vague), False)
    passed &= check(
        "other 但没填名字被拦",
        passes_gate(item(user_specific=False, subject="other", subject_name=None)),
        False,
    )

    print("\n[5] 有效期")
    today = date(2026, 10, 31)
    passed &= check("昨天到期 → 已过期", is_expired("2026-10-30", today), True)
    passed &= check("今天到期 → 仍可用", is_expired("2026-10-31", today), False)
    passed &= check("明年到期 → 仍可用", is_expired("2027-06-07", today), False)
    passed &= check("空值 → 永不过期", is_expired(None, today), False)
    passed &= check("格式错 → 不误删", is_expired("下个月", today), False)

    print("\n" + ("全部通过" if passed else "有失败项，见上面的 ✗"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
