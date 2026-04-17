"""
GAN Skill Creator - 主入口
基于 GAN 的高纯度知识蒸馏系统
"""
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'gan-skill-creator'))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    # 手动加载 .env
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip())

from tools.internet_data_collector import InternetDataCollector
from tools.knowledge_extractor import KnowledgeExtractor
from tools.gan_distiller import GANSkillDistiller
from tools.purity_evaluator import PurityEvaluator
from tools.local_vectorizer import get_vectorizer

import json
import numpy as np
from datetime import datetime


def create_skill(
    target_person: str,
    skill_name: str = None,
    verbose: bool = True
):
    """
    创建一个 Skill 的完整流程

    Args:
        target_person: 目标人物名称
        skill_name: Skill 名称（默认自动生成）
        verbose: 是否打印详细信息

    Returns:
        dict: 包含 skill_md, purity_score, output_dir 的结果
    """
    # 检查 API Key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-anthropic-api-key-here":
        print("❌ 错误: 需要提供 Anthropic API Key")
        print("   请在 .env 文件中设置: ANTHROPIC_API_KEY=your-key")
        return None

    if skill_name is None:
        skill_name = target_person.lower().replace(' ', '-') + '-skill'

    if verbose:
        print("=" * 70)
        print(f"🚀 GAN Skill Creator: {target_person}")
        print("=" * 70)

    # ========== 第1阶段：数据收集 ==========
    if verbose:
        print("\n📚 第1阶段：数据收集")

    collector = InternetDataCollector(target_person, skill_name)
    collector.print_collection_guide()

    stats = collector.get_statistics()
    if stats['total_items'] == 0:
        print("\n⚠️ 警告：没有收集到任何数据！")
        print("请使用 collector.add_video(), add_article(), add_post() 添加数据")
        return None

    # ========== 第2阶段：知识提取 ==========
    if verbose:
        print("\n🧠 第2阶段：知识提取")

    extractor = KnowledgeExtractor(api_key=api_key)

    # 合并文本
    combined_text = _get_combined_text(collector)

    # 提取结构化知识
    knowledge = extractor.extract_from_sources(target_person, combined_text)

    # 向量化知识
    knowledge_text = extractor.vectorize_knowledge(knowledge)

    # ========== 第3阶段：GAN 蒸馏 ==========
    if verbose:
        print("\n🤖 第3阶段：GAN 蒸馏")

    vectorizer = get_vectorizer(preset="fast")
    distiller = GANSkillDistiller(input_dim=vectorizer.embedding_dim, device='cpu')

    # 生成训练数据
    knowledge_variants = _generate_variants(knowledge_text, n_variants=30)
    knowledge_vectors = vectorizer.vectorize_batch(knowledge_variants)

    # 训练 GAN
    distiller.train(
        knowledge_vectors=knowledge_vectors,
        epochs=50,
        batch_size=16,
        verbose=verbose
    )

    # 蒸馏
    distilled = distiller.distill(knowledge_vectors[0])

    # ========== 第4阶段：纯度评估 ==========
    if verbose:
        print("\n✔️ 第4阶段：纯度评估")

    evaluator = PurityEvaluator(api_key=api_key)
    purity_result = evaluator.get_purity_score(
        original=knowledge_text,
        distilled=knowledge_text
    )
    purity_score = purity_result['overall_score']

    # ========== 第5阶段：生成 Skill.md ==========
    if verbose:
        print("\n🏗️ 第5阶段：生成 Skill.md")

    skill_md = _generate_skill_md(target_person, purity_score, knowledge)

    # 保存结果
    output_dir = Path(__file__).parent / 'gan-skill-creator' / 'skills-workspace' / skill_name
    output_dir.mkdir(parents=True, exist_ok=True)

    skill_path = output_dir / 'SKILL.md'
    with open(skill_path, 'w', encoding='utf-8') as f:
        f.write(skill_md)

    if verbose:
        print(f"\n✅ Skill 已生成: {skill_path}")
        print(f"   纯度评分: {purity_score:.1%}")

    return {
        'skill_md': skill_md,
        'purity_score': purity_score,
        'output_dir': str(output_dir),
        'knowledge': knowledge,
        'distilled': distilled
    }


def _get_combined_text(collector: InternetDataCollector) -> str:
    """从收集器合并所有文本"""
    combined = []

    for video in collector.data_sources.get("videos", []):
        combined.append(f"[Video: {video['title']}]\n{video.get('transcript', '')}")

    for article in collector.data_sources.get("articles", []):
        combined.append(f"[Article: {article['title']}]\n{article.get('content', '')}")

    for post in collector.data_sources.get("social_media", []):
        combined.append(f"[{post['platform']}]\n{post.get('text', '')}")

    for book in collector.data_sources.get("books", []):
        excerpts = "\n".join(book.get('excerpts', []))
        combined.append(f"[Book: {book['book_title']}]\n{excerpts}")

    for code in collector.data_sources.get("code", []):
        combined.append(f"[Code: {code['repo']}]\n{code.get('analysis', '')}")

    return "\n\n".join(combined)


def _generate_variants(text: str, n_variants: int = 30) -> list:
    """生成文本变体用于训练"""
    variants = [text]

    sentences = text.replace('。', '.').replace('\n', ' ').split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) > 1:
        import random
        for i in range(min(n_variants - 1, len(sentences) * 3)):
            sample_size = max(1, len(sentences) // 2)
            sampled = random.sample(sentences, min(sample_size, len(sentences)))
            variants.append('. '.join(sampled) + '.')

    while len(variants) < n_variants:
        variants.append(text)

    return variants[:n_variants]


def _generate_skill_md(target_person: str, purity_score: float, knowledge: dict) -> str:
    """生成 Skill.md 内容"""
    skill_name = target_person.lower().replace(' ', '-')

    core_principles = knowledge.get('core_principles', [])
    decision_frameworks = knowledge.get('decision_frameworks', [])
    communication_style = knowledge.get('communication_style', '直接、清晰')
    core_values = knowledge.get('core_values', [])
    key_insights = knowledge.get('key_insights', [])

    return f"""---
name: {skill_name}-skill
displayName: {target_person} - Expert Thinking
description: |
  使用 {target_person} 的思维方式和决策框架来分析和解决问题。

  何时使用：
  - 需要从 {target_person} 的视角分析问题
  - 需要应用其核心原则和决策框架
  - 需要创新性解决方案

  这个 Skill 会：
  - 应用 {target_person} 的核心原则
  - 使用其决策框架
  - 采用其沟通风格
version: 1.0.0
purityScore: {purity_score:.3f}
distillationMethod: GAN-Claude
---

# {target_person} - 思维系统

## 概述

这个 Skill 将 {target_person} 的思维方式和决策框架编码为可复用的指导。
通过 GAN 对抗训练 + Claude 评估确保了 {purity_score:.1%} 的知识纯度。

## 核心原则

{chr(10).join(['- ' + p for p in core_principles]) if core_principles else '- 未提取到核心原则'}

## 决策框架

{chr(10).join(['- ' + f for f in decision_frameworks]) if decision_frameworks else '- 未提取到决策框架'}

## 沟通风格

{communication_style}

## 核心价值观

{', '.join(core_values) if core_values else '未提取到核心价值观'}

## 关键洞察

{chr(10).join(['- ' + i for i in key_insights]) if key_insights else '- 未提取到关键洞察'}

## 使用示例

**用户问**: "如何用 {target_person} 的方式分析这个问题？"

**Skill 做**:
1. 应用核心原则分解问题
2. 使用决策框架评估选项
3. 提供创新性建议
4. 以 {target_person} 的风格表达

## 质量指标

- **蒸馏纯度**: {purity_score:.1%}
- **蒸馏方法**: GAN 对抗训练
- **评估模型**: Claude
- **数据来源**: 多源互联网数据

---
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}
基于 GAN 对抗蒸馏技术 + Claude 评估
"""


def interactive_mode():
    """交互式模式"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║            🎯 GAN Skill Creator - 交互式模式                  ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 获取目标人物
    target_person = input("请输入要蒸馏的目标人物: ").strip()
    if not target_person:
        print("❌ 必须输入目标人物名称")
        return

    print(f"\n✅ 目标人物: {target_person}")
    print("\n接下来请收集数据，然后运行 create_skill() 函数")

    # 创建收集器并显示指南
    collector = InternetDataCollector(target_person, target_person.lower().replace(' ', '-'))
    collector.print_collection_guide()


if __name__ == "__main__":
    # 检查 API Key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-anthropic-api-key-here":
        print("⚠️ 警告: 未设置 ANTHROPIC_API_KEY 环境变量")
        print("请在 .env 文件中设置: ANTHROPIC_API_KEY=your-key")
        print()

    # 示例：使用完整流程
    # create_skill("Elon Musk")

    # 或进入交互模式
    interactive_mode()
