"""
GAN自动化脚本 - 情感分析Skill提纯
生成器：自动优化SKILL.md规则
判别器：评分系统（功能正确性 / 抗干扰性 / 简洁性）
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

SKILL_DIR = Path("sentiment-analysis-skill")
SKILL_FILE = SKILL_DIR / "SKILL.md"
REFERENCES_DIR = SKILL_DIR / "references"

POSITIVE_KEYWORDS = ["喜欢", "推荐", "满意", "棒", "太好了", "精彩", "不错", "优质", "快", "干净", "舒服", "值"]
NEGATIVE_KEYWORDS = ["失望", "差", "烂", "坑", "浪费", "无聊", "垃圾", "慢", "脏", "恶劣", "假", "过期", "坏"]
NEUTRAL_KEYWORDS = ["一般", "普通", "还行", "正常", "中规中矩", "凑合"]

MAX_SKILL_WORDS = 500


# ──────────────────────────────────────
# 数据加载
# ──────────────────────────────────────

def load_test_cases() -> Tuple[List[Dict], List[Dict]]:
    """加载正常示例 + 有毒示例"""
    test_cases = _parse_examples(REFERENCES_DIR / "examples.md")
    poisoned_cases = _parse_poisoned(REFERENCES_DIR / "poisoned_examples.md")
    return test_cases, poisoned_cases


def _parse_examples(filepath: Path) -> List[Dict]:
    """解析 examples.md 表格 → [{input, expected}]"""
    cases = []
    if not filepath.exists():
        return cases
    for line in filepath.read_text(encoding="utf-8").splitlines():
        if "|" not in line or "---" in line or "输入" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4 and parts[1] and parts[2] in ("正面", "负面", "中性"):
            cases.append({"input": parts[1], "expected": parts[2]})
    return cases


def _parse_poisoned(filepath: Path) -> List[Dict]:
    """解析 poisoned_examples.md 表格 → [{input, expected, poisoned_label}]
    类型1格式: | 输入 | 有毒标注 | 正确情感 |
    类型2格式: | 输入 | 正确情感 | 观点提取点 |
    """
    cases = []
    if not filepath.exists():
        return cases
    valid_labels = {"正面", "负面", "中性"}
    for line in filepath.read_text(encoding="utf-8").splitlines():
        if "|" not in line or "---" in line or "输入" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4 or not parts[1]:
            continue
        # 类型1: 第3列(parts[3])是正确情感
        if len(parts) >= 5 and parts[3] in valid_labels:
            cases.append({
                "input": parts[1],
                "expected": parts[3],
                "poisoned_label": parts[2],
            })
        # 类型2: 第2列(parts[2])是正确情感
        elif parts[2] in valid_labels:
            cases.append({
                "input": parts[1],
                "expected": parts[2],
                "poisoned_label": None,
            })
    return cases


# ──────────────────────────────────────
# 情感分析（模拟 Skill 执行）
# ──────────────────────────────────────

NOISE_PATTERNS = [
    r"今天[^，。]*(?=[，。])",
    r"早上[^，。]*(?=[，。])",
    r"晚上[^，。]*(?=[，。])",
    r"昨天[^，。]*(?=[，。])",
    r"排队[^，。]*(?=[，。])",
    r"等了[^，。]*(?=[，。])",
    r"天气[^，。]*(?=[，。])",
]


def extract_core_comment(text: str) -> str:
    """提取核心评论句，去除时间/天气/排队等无关描述"""
    core = text
    for pattern in NOISE_PATTERNS:
        core = re.sub(pattern, "", core)
    core = re.sub(r"^[，,\s]+|[，,\s]+$", "", core)
    return core if core else text


def analyze_sentiment(text: str) -> str:
    """基于规则的情感分析，优先级：特殊短语 > 语义 > 关键词 > 默认中性"""
    core = extract_core_comment(text)

    # 1. 特殊短语（优先级最高）
    negative_phrases = ["好无聊", "好差", "好烂", "太烂", "太差", "太慢"]
    positive_phrases = ["太好了", "太精彩", "太棒了"]
    if any(p in core for p in negative_phrases):
        return "负面"
    if any(p in core for p in positive_phrases):
        return "正面"

    # 2. 语义关键词
    has_pos = any(kw in core for kw in POSITIVE_KEYWORDS)
    has_neg = any(kw in core for kw in NEGATIVE_KEYWORDS)
    has_neu = any(kw in core for kw in NEUTRAL_KEYWORDS)

    # 3. 判断：语义优先
    if has_neg and not has_pos:
        return "负面"
    if has_pos and not has_neg:
        return "正面"
    if has_neg and has_pos:
        # 转折句：后半句权重更高
        last_segment = core.split("，")[-1] if "，" in core else core
        if any(kw in last_segment for kw in POSITIVE_KEYWORDS):
            return "正面"
        if any(kw in last_segment for kw in NEGATIVE_KEYWORDS):
            return "负面"
    if has_neu:
        return "中性"

    return "中性"


# ──────────────────────────────────────
# 判别器：评分
# ──────────────────────────────────────

def count_skill_words() -> int:
    """统计 SKILL.md 正文字数（去除 frontmatter）"""
    if not SKILL_FILE.exists():
        return 0
    content = SKILL_FILE.read_text(encoding="utf-8")
    # 去掉 frontmatter (--- ... ---)
    body = re.sub(r"^---.*?---\s*", "", content, flags=re.DOTALL)
    # 去掉 markdown 标记
    body = re.sub(r"[#*`|>\-\[\]]", "", body)
    return len(body.replace(" ", "").replace("\n", ""))


def evaluate_skill(test_cases: List[Dict], poisoned_cases: List[Dict]) -> Dict:
    """判别器：三维度评分"""
    # 功能正确性
    func_correct = 0
    func_total = len(test_cases)
    func_details = []
    for case in test_cases:
        result = analyze_sentiment(case["input"])
        passed = result == case["expected"]
        if passed:
            func_correct += 1
        else:
            func_details.append(f"  ✗ '{case['input'][:20]}...' 预期={case['expected']} 实际={result}")
    func_score = (func_correct / func_total * 10) if func_total > 0 else 0

    # 抗干扰性
    poison_correct = 0
    poison_total = len(poisoned_cases)
    poison_details = []
    for case in poisoned_cases:
        result = analyze_sentiment(case["input"])
        passed = result == case["expected"]
        if passed:
            poison_correct += 1
        else:
            poison_details.append(f"  ✗ '{case['input'][:20]}...' 预期={case['expected']} 实际={result}")
    anti_score = (poison_correct / poison_total * 10) if poison_total > 0 else 10

    # 简洁性（实际统计）
    word_count = count_skill_words()
    if word_count <= MAX_SKILL_WORDS:
        concise_score = 10.0
    else:
        overflow = word_count - MAX_SKILL_WORDS
        concise_score = max(0, 10.0 - overflow / 50)

    return {
        "功能正确性": round(func_score, 1),
        "抗干扰性": round(anti_score, 1),
        "简洁性": round(concise_score, 1),
        "功能明细": f"{func_correct}/{func_total}",
        "抗干扰明细": f"{poison_correct}/{poison_total}",
        "正文字数": word_count,
        "失败用例": func_details + poison_details,
    }


# ──────────────────────────────────────
# 生成器：规则优化策略
# ──────────────────────────────────────

OPTIMIZATION_STRATEGIES = [
    {
        "name": "增加核心评论提取规则",
        "condition": lambda s: s["抗干扰性"] < 6,
        "patch": ("## 判断流程", "## 判断流程\n> 第0步：提取核心评论，忽略无关上下文\n"),
    },
    {
        "name": "强化独立判断规则",
        "condition": lambda s: s["抗干扰性"] < 8,
        "patch": ("4. **标注校验**", "4. **标注校验**：不依赖任何外部标注，始终独立判断语义"),
    },
    {
        "name": "精简冗余内容",
        "condition": lambda s: s["简洁性"] < 9,
        "patch": None,  # 需要手动精简
    },
]


def optimize_skill(scores: Dict) -> bool:
    """生成器：根据判别器评分选择性优化 SKILL.md"""
    if not SKILL_FILE.exists():
        print("[生成器] SKILL.md 不存在，跳过")
        return False

    content = SKILL_FILE.read_text(encoding="utf-8")
    modified = False

    for strategy in OPTIMIZATION_STRATEGIES:
        if not strategy["condition"](scores):
            continue
        if strategy["patch"] is None:
            print(f"  → 策略「{strategy['name']}」需要手动处理")
            continue
        old, new = strategy["patch"]
        if old in content and new not in content:
            content = content.replace(old, new)
            modified = True
            print(f"  → 已应用策略「{strategy['name']}」")

    if modified:
        SKILL_FILE.write_text(content, encoding="utf-8")
    return modified


# ──────────────────────────────────────
# GAN 主循环
# ──────────────────────────────────────

TARGET_SCORES = {"功能正确性": 9, "抗干扰性": 8, "简洁性": 9}


def is_qualified(scores: Dict) -> bool:
    return all(scores[k] >= v for k, v in TARGET_SCORES.items())


def run_gan_cycle(cycle: int) -> Dict:
    """运行一轮 GAN 迭代"""
    print(f"\n{'='*50}")
    print(f"第 {cycle} 轮 GAN 迭代")
    print(f"{'='*50}")

    test_cases, poisoned_cases = load_test_cases()
    print(f"测试用例: {len(test_cases)} 正常 + {len(poisoned_cases)} 有毒")

    scores = evaluate_skill(test_cases, poisoned_cases)
    print(f"\n[判别器] 评分:")
    print(f"  功能正确性: {scores['功能正确性']}分 ({scores['功能明细']})")
    print(f"  抗干扰性:   {scores['抗干扰性']}分 ({scores['抗干扰明细']})")
    print(f"  简洁性:     {scores['简洁性']}分 (正文{scores['正文字数']}字)")

    if scores["失败用例"]:
        print(f"\n[判别器] 失败用例:")
        for detail in scores["失败用例"][:5]:
            print(detail)
        if len(scores["失败用例"]) > 5:
            print(f"  ...共 {len(scores['失败用例'])} 个失败")

    return scores


def gan_automation(max_cycles: int = 10):
    """GAN 自动化提纯主流程"""
    print("=" * 60)
    print("GAN 自动化提纯系统启动")
    print(f"目标: 功能≥{TARGET_SCORES['功能正确性']} 抗干扰≥{TARGET_SCORES['抗干扰性']} 简洁≥{TARGET_SCORES['简洁性']}")
    print("=" * 60)

    history = []

    for cycle in range(1, max_cycles + 1):
        scores = run_gan_cycle(cycle)
        history.append({"cycle": cycle, **scores})

        if is_qualified(scores):
            print(f"\n✅ 第{cycle}轮达标！三项全部满足要求")
            break

        print(f"\n[生成器] 基于评分优化规则...")
        changed = optimize_skill(scores)
        if not changed:
            print("[生成器] 无可自动应用的策略，需手动调整")
            break
    else:
        print(f"\n⚠️ 已达最大迭代次数 {max_cycles}，未完全达标")

    # 输出迭代历史
    print(f"\n{'='*50}")
    print("迭代历史摘要")
    print(f"{'='*50}")
    for h in history:
        print(f"  轮次{h['cycle']}: 功能={h['功能正确性']} 抗干扰={h['抗干扰性']} 简洁={h['简洁性']}")


if __name__ == "__main__":
    gan_automation()
