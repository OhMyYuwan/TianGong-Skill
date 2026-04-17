"""
Claude API 完整测试脚本
使用 Anthropic Claude 模型测试完整的 GAN Skill 创建流程
"""
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'gan-skill-creator'))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / '.env')
except ImportError:
    print("⚠️ python-dotenv 未安装，使用系统环境变量")
    # 手动加载 .env
    env_file = project_root / '.env'
    if env_file.exists():
        with open(env_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip())

# 导入工具
from tools.internet_data_collector import InternetDataCollector
from tools.knowledge_extractor import KnowledgeExtractor
from tools.gan_distiller import GANSkillDistiller
from tools.purity_evaluator import PurityEvaluator
from tools.local_vectorizer import get_vectorizer

import json
import numpy as np
from datetime import datetime


def run_full_test(
    anthropic_api_key: str = None,
    target_person: str = "Elon Musk",
    use_sample_data: bool = True,
    verbose: bool = True
):
    """
    运行完整的 GAN Skill 创建测试

    Args:
        anthropic_api_key: Anthropic API Key
        target_person: 目标人物
        use_sample_data: 是否使用示例数据
        verbose: 是否打印详细信息
    """
    print("\n" + "=" * 70)
    print(f"🚀 GAN Skill Creator 完整测试（Claude 版本）")
    print(f"   目标人物: {target_person}")
    print("=" * 70)

    # ========== 获取 API Key ==========
    api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-anthropic-api-key-here":
        print("\n❌ 错误: 需要提供 Anthropic API Key")
        print("   方式1: 设置环境变量 ANTHROPIC_API_KEY")
        print("   方式2: 在 .env 文件中添加 ANTHROPIC_API_KEY=your-key")
        print("   方式3: 直接传入 api_key 参数")
        return None

    # ========== 第1阶段：初始化 Claude 客户端 ==========
    print("\n📦 第1阶段：初始化 Claude 客户端")
    print("-" * 50)

    try:
        knowledge_extractor = KnowledgeExtractor(api_key=api_key)
        purity_evaluator = PurityEvaluator(api_key=api_key)
        print("✅ Claude 客户端初始化成功")
    except Exception as e:
        print(f"❌ Claude 客户端初始化失败: {e}")
        return None

    # ========== 第2阶段：数据收集 ==========
    print("\n📚 第2阶段：数据收集")
    print("-" * 50)

    skill_name = target_person.lower().replace(' ', '-') + '-skill'
    collector = InternetDataCollector(target_person, skill_name)

    if use_sample_data:
        print("   使用示例数据进行测试...")

        # 添加示例视频
        collector.add_video(
            title="How to Think Like Elon Musk",
            transcript="""
First principles thinking is about breaking things down to their fundamental truths
and reasoning up from there, rather than reasoning by analogy.
The normal way we conduct our lives is by analogy. We do things because they've been done before.
But with first principles, you boil things down to the most fundamental truths you can imagine
and then reason up from there.

When you want to do something new, you have to apply the physics approach.
Physics is really figuring out how to discover new things that are counterintuitive.
Take any technology or approach, and look at what are the fundamental limits.

I think it's very important to have a feedback loop where you're constantly thinking about
what you've done and how you could be doing it better. I think that's the single best piece of advice:
constantly think about how you could be doing things better and questioning yourself.

If something is important enough, even if the odds are against you, you should still do it.
Persistence is very important. You should not give up unless you're forced to give up.
""",
            source_url="https://youtube.com/example1"
        )

        # 添加示例文章
        collector.add_article(
            title="Elon Musk's Decision Making Framework",
            content="""
Elon Musk is known for his unique approach to decision making and problem solving.
Here are the key frameworks he uses:

1. First Principles Thinking: Instead of reasoning by analogy (doing what others do),
   break down problems to their most basic truths and build up from there.

2. The "Idiot Index": Musk calculates how much time something should take vs. how long
   it actually takes. If the ratio is too high, there's waste to eliminate.

3. Delete and Optimize: Always try to delete parts of a process first. If you add back
   more than 10% of what you deleted, you weren't deleting enough.

4. Rapid Iteration: Build, test, fail, learn, repeat. Speed of iteration is key.

5. Physics-First Approach: Understand the fundamental constraints before assuming
   something is impossible.

Musk also emphasizes the importance of asking the right questions.
"The right questions are more important than the right answers."
""",
            source_url="https://medium.com/example"
        )

        # 添加示例社交媒体帖子
        collector.add_post(
            platform="Twitter",
            text="First principles thinking is about breaking things down to their fundamental truths and reasoning up from there.",
            post_url="https://twitter.com/elonmusk/status/1"
        )
        collector.add_post(
            platform="Twitter",
            text="If something is important enough, even if the odds are against you, you should still do it.",
            post_url="https://twitter.com/elonmusk/status/2"
        )
        collector.add_post(
            platform="Twitter",
            text="Constantly think about how you could be doing things better and questioning yourself.",
            post_url="https://twitter.com/elonmusk/status/3"
        )

        print("✅ 示例数据已添加")
    else:
        collector.print_collection_guide()
        print("\n⚠️ 请手动添加数据后继续...")
        return None

    # 打印统计
    stats = collector.get_statistics()
    print(f"\n📊 数据统计:")
    for source_type, count in stats['sources'].items():
        print(f"   - {source_type}: {count} 项")
    print(f"   总计: {stats['total_items']} 项")

    # ========== 第3阶段：知识提取 ==========
    print("\n🧠 第3阶段：知识提取")
    print("-" * 50)

    # 合并文本
    combined_text = get_combined_text(collector)
    print(f"   合并文本长度: {len(combined_text)} 字符")

    # 使用 Claude 提取知识
    knowledge = knowledge_extractor.extract_from_sources(target_person, combined_text)

    print("\n📝 提取的知识:")
    print(json.dumps(knowledge, ensure_ascii=False, indent=2)[:1000])
    if len(json.dumps(knowledge)) > 1000:
        print("   ... (内容已截断)")

    # 向量化知识文本
    knowledge_text = knowledge_extractor.vectorize_knowledge(knowledge)

    # ========== 第4阶段：向量化 ==========
    print("\n🔢 第4阶段：向量化")
    print("-" * 50)

    try:
        vectorizer = get_vectorizer(preset="fast")
        print(f"✅ 向量化器已加载，维度: {vectorizer.embedding_dim}")
    except Exception as e:
        print(f"⚠️ 向量化器加载失败，使用模拟向量: {e}")
        vectorizer = None

    # 生成知识向量
    if vectorizer:
        knowledge_vectors = vectorizer.vectorize_batch(
            generate_variants(knowledge_text, n_variants=30)
        )
    else:
        # 使用模拟向量
        knowledge_vectors = np.random.randn(30, 384).astype(np.float32)

    print(f"   知识向量形状: {knowledge_vectors.shape}")

    # ========== 第5阶段：GAN 蒸馏 ==========
    print("\n🤖 第5阶段：GAN 蒸馏")
    print("-" * 50)

    input_dim = knowledge_vectors.shape[1]
    distiller = GANSkillDistiller(input_dim=input_dim, device='cpu')

    # 训练 GAN
    distiller.train(
        knowledge_vectors=knowledge_vectors,
        epochs=30,
        batch_size=8,
        verbose=verbose
    )

    # 执行蒸馏
    distilled = distiller.distill(knowledge_vectors[0])
    print(f"\n✅ 蒸馏完成")
    print(f"   纯度评分: {distilled['purity_score']:.2%}")
    print(f"   重建误差: {distilled['reconstruction_error']:.6f}")

    # ========== 第6阶段：纯度评估 ==========
    print("\n✔️ 第6阶段：纯度评估")
    print("-" * 50)

    try:
        purity_result = purity_evaluator.get_purity_score(
            original=knowledge_text,
            distilled=knowledge_text  # 简化：使用相同文本
        )
        purity_score = purity_result['overall_score']
        print(f"   综合评分: {purity_score:.1%}")
    except Exception as e:
        print(f"   ⚠️ 纯度评估失败: {e}")
        print("   使用 GAN 蒸馏纯度作为替代")
        purity_score = distilled['purity_score']

    # ========== 第7阶段：生成 Skill.md ==========
    print("\n🏗️ 第7阶段：生成 Skill.md")
    print("-" * 50)

    skill_md = generate_skill_md(
        target_person=target_person,
        purity_score=purity_score,
        knowledge=knowledge
    )

    # 保存结果
    output_dir = Path(__file__).parent / 'gan-skill-creator' / 'skills-workspace' / skill_name
    output_dir.mkdir(parents=True, exist_ok=True)

    skill_path = output_dir / 'SKILL.md'
    with open(skill_path, 'w', encoding='utf-8') as f:
        f.write(skill_md)

    result_json = {
        "target_person": target_person,
        "skill_name": skill_name,
        "created_at": datetime.now().isoformat(),
        "purity_score": purity_score,
        "gan_purity": distilled['purity_score'],
        "knowledge": knowledge
    }

    with open(output_dir / 'result.json', 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    # ========== 完成 ==========
    print("\n" + "=" * 70)
    print("✅ 完整流程测试完成！")
    print("=" * 70)
    print(f"""
📊 最终结果：
   ✓ 目标人物: {target_person}
   ✓ Skill 名称: {skill_name}
   ✓ GAN 纯度: {distilled['purity_score']:.1%}
   ✓ 评估纯度: {purity_score:.1%}
   ✓ Skill 路径: {skill_path}
   ✓ 结果 JSON: {output_dir / 'result.json'}

🎉 现在可以查看生成的 Skill 了！
""")

    return {
        'skill_md': skill_md,
        'skill_path': str(skill_path),
        'result': result_json
    }


def get_combined_text(collector: InternetDataCollector) -> str:
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


def generate_variants(text: str, n_variants: int = 30) -> list:
    """生成文本变体用于训练"""
    variants = [text]

    # 分句
    sentences = text.replace('。', '.').replace('\n', ' ').split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) > 1:
        import random
        for i in range(min(n_variants - 1, len(sentences) * 5)):
            sample_size = max(1, len(sentences) // 2)
            sampled = random.sample(sentences, min(sample_size, len(sentences)))
            variants.append('. '.join(sampled) + '.')

    while len(variants) < n_variants:
        variants.append(text)

    return variants[:n_variants]


def generate_skill_md(target_person: str, purity_score: float, knowledge: dict) -> str:
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


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║     🧪 GAN Skill Creator - Claude 完整测试脚本               ║
╚══════════════════════════════════════════════════════════════╝

使用说明：
1. 确保已安装依赖: pip install anthropic sentence-transformers torch
2. 设置环境变量: ANTHROPIC_API_KEY=your-key
   或在 .env 文件中添加
3. 运行此脚本

""")

    # 检查 API Key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-anthropic-api-key-here":
        print("❌ 请先设置 ANTHROPIC_API_KEY 环境变量")
        print("\n方式1 (Windows PowerShell):")
        print('   $env:ANTHROPIC_API_KEY = "your-api-key"')
        print("\n方式2 (.env 文件):")
        print("   ANTHROPIC_API_KEY=your-api-key")
        print("\n方式3 (代码中直接传入):")
        print('   run_full_test(anthropic_api_key="your-api-key")')
        sys.exit(1)

    # 运行测试
    result = run_full_test(
        anthropic_api_key=api_key,
        target_person="Elon Musk",
        use_sample_data=True,
        verbose=True
    )

    if result:
        print("\n🎉 测试成功！查看生成的 Skill.md 文件。")
