"""Step 4：合规拦截层 —— 分级规则 + 产品语境感知

核心思想：合规红线不是「出现了某个词」，而是「在错误的语境下出现了某个词」。

- 存款类产品（定期/活期/大额存单）：受《存款保险条例》保护，
  「保本保息」是法定事实，可以讲、甚至应该讲。
- 理财/基金/资管类产品：净值型、打破刚兑，禁止承诺「保本」「无风险」「稳赚」。
- 绝对化用语（稳赚不赔 / 绝对无风险 / 保证收益 X%）：任何产品都禁止，违反《广告法》。

因此规则分两级：
  🔴 红灯词 —— 任何语境都拦（绝对化承诺）；
  🟡 黄灯词 —— 只有落在「非存款语境」才拦（如理财讲保本）；
             语境不明时不硬拦，交给 LLM 软审查，避免误伤。
"""
import re

# ============ 第一级：红灯词（任何产品都不能出现，无条件硬拦截）============
RED_PATTERNS = [
    (r"稳赚不赔|稳赚|必赚|只赚不赔|包赚|躺赚", "承诺收益（绝对化用语）"),
    (r"保证.{0,4}收益|承诺.{0,4}收益|确保.{0,4}收益|收益.{0,4}保证", "承诺收益"),
    (r"绝对无风险|零风险|毫无风险|没有任何风险", "承诺无风险（绝对化用语）"),
]

# ============ 第二级：黄灯词（需结合产品语境判断）============
AMBER_PATTERNS = [
    (r"保本保息|保本|保底|本金无忧", "保本表述"),
    (r"无风险", "无风险表述"),
]

# 存款语境标志词（出现 → 是存款类产品，「保本保息」合法）
DEPOSIT_HINTS = re.compile(r"存款|存单|定期|活期|大额存单|存款利率|存款保险")
# 非存款（资管类）语境标志词（出现 → 禁止「保本 / 无风险」）
NON_DEPOSIT_HINTS = re.compile(r"理财|基金|资管|净值|权益类|混合类|固收类|净值型")


def rule_check(text: str):
    """分级 + 语境感知的规则检查，返回命中原因列表。"""
    hits = []

    # 1) 红灯：无条件拦截
    for pattern, reason in RED_PATTERNS:
        if re.search(pattern, text):
            hits.append(reason)

    # 2) 黄灯：结合产品语境判断
    for pattern, reason in AMBER_PATTERNS:
        if not re.search(pattern, text):
            continue
        has_non_deposit = bool(NON_DEPOSIT_HINTS.search(text))
        has_deposit = bool(DEPOSIT_HINTS.search(text))
        if has_non_deposit:
            # 提到理财/基金/资管 → 绝不能讲保本或无风险
            hits.append(f"{reason}（理财/基金等资管产品不得承诺保本或无风险）")
        elif not has_deposit:
            # 既没提存款也没提理财，语境不明 → 不硬拦，交给 LLM 软审查
            continue
        # else：纯存款语境 → 合法，放行

    return hits


def compliance_check(text: str):
    """返回 (是否通过, 命中原因列表)。命中任一违规项即不通过。"""
    hits = rule_check(text)
    return (len(hits) == 0), hits


if __name__ == "__main__":
    # (测试文本, 预期是否通过)
    tests = [
        ("这款理财产品保本保息，绝对无风险，放心买。", False),          # 理财+保本 → 拦
        ("R2 理财稳赚不赔，本金无忧。", False),                        # 绝对化用语 → 拦
        ("该产品净值型运作，收益保本，可放心购买。", False),            # 理财讲保本 → 拦
        ("定期存款受《存款保险条例》保护，属于保本保息类存款产品。", True),   # 纯存款 → 放行
        ("大额存单保本保息，3 年期利率 2.60%，高于普通定期存款。", True),   # 存款比较 → 放行
        ("该产品业绩比较基准为 2.5%-3.5%，不代表实际收益，不构成收益承诺。", True),  # 合规话术 → 放行
        ("该产品可能保本。", True),                                    # 语境不明 → 不硬拦，交 LLM
        # —— 用户真实场景：问「有没有更高利率的存款」 ——
        ("6 个月定期存款挂牌利率为 1.30%。如需更高利率，可选择 3 年期 1.95%，"
         "或大额存单 3 年期 2.60%（存款类，受存款保险保障、保本保息）。", True),
    ]
    for text, expect in tests:
        ok, hits = compliance_check(text)
        flag = "✅ 通过" if ok else "❌ 拦截"
        mark = "OK" if ok == expect else "!! 与预期不符"
        print(f"{flag} [{mark}]：{text}  →  {hits}")
