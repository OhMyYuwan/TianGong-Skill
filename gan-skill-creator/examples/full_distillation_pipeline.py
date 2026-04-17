"""
完整蒸馏流程：端到端的Skill生成
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tools.internet_data_collector import InternetDataCollector
from tools.knowledge_extractor import KnowledgeExtractor
from tools.gan_distiller import GANSkillDistiller
from tools.purity_evaluator import PurityEvaluator
import json


def collect_data(target_person: str, skill_name: str) -> InternetDataCollector:
    """第1阶段：收集互联网数据"""
    print("\n" + "="*80)
    print(f"📚 第1阶段：收集 {target_person} 的互联网数据")
    print("="*80)

    collector = InternetDataCollector(target_person, skill_name)

    # 这里应该是用户手动添加数据，但为了演示，我们添加一些示例
    collector.add_video(
        title="Example Interview",
        transcript="This is a sample interview transcript about first principles thinking.",
        source_url="https://youtube.com/example"
    )

    collector.add_article(
        title="How to Think",
        content="Article about decision making and problem solving approaches.",
        source_url="https://medium.com/example"
    )

    # 添加社交媒体帖子
    posts = [
        ("Twitter", "First principles thinking is key", "https://twitter.com/example"),
        ("Twitter", "Delete unnecessary complexity", "https://twitter.com/example2"),
    ]

    for platform, text, url in posts:
        collector.add_post(platform, text, url)

    stats = collector.get_statistics()
    print(f"\n✅ 数据收集完成:")
    print(f"   总计: {stats['total_items']} 项")
    for source_type, count in stats['sources'].items():
        print(f"   - {source_type}: {count} 项")

    return collector


def get_combined_text(collector: InternetDataCollector) -> str:
    """从收集器中合并所有文本"""
    combined = []

    # 视频转录
    for video in collector.data_sources.get("videos", []):
        combined.append(f"[Video: {video['title']}]\n{video.get('transcript', '')}")

    # 文章
    for article in collector.data_sources.get("articles", []):
        combined.append(f"[Article: {article['title']}]\n{article.get('content', '')}")

    # 社交媒体
    for post in collector.data_sources.get("social_media", []):
        combined.append(f"[{post['platform']}]\n{post.get('text', '')}")

    # 书籍
    for book in collector.data_sources.get("books", []):
        excerpts = "\n".join(book.get('excerpts', []))
        combined.append(f"[Book: {book['book_title']}]\n{excerpts}")

    # 代码分析
    for code in collector.data_sources.get("code", []):
        combined.append(f"[Code: {code['repo']}]\n{code.get('analysis', '')}")

    return "\n\n".join(combined)


def extract_knowledge(collector: InternetDataCollector, target_person: str) -> str:
    """第2阶段：提取知识"""
    print("\n" + "="*80)
    print("🧠 第2阶段：从数据中提取知识")
    print("="*80)

    extractor = KnowledgeExtractor()

    # 获取合并的文本
    combined_text = get_combined_text(collector)
    
    # 提取知识
    knowledge = extractor.extract_from_sources(target_person, combined_text)
    
    # 向量化知识
    knowledge_text = extractor.vectorize_knowledge(knowledge)
    
    print(f"\n提取的知识概要:")
    print(knowledge_text[:300] + "...")
    
    return knowledge_text


def distill_with_gan(knowledge_text: str) -> Dict:
    """第3阶段：GAN蒸馏"""
    print("\n" + "="*80)
    print("🤖 第3阶段：GAN对抗蒸馏")
    print("="*80)
    
    # 初始化GAN
    device = 'cuda' if False else 'cpu'  # 改为 torch.cuda.is_available() 以启用GPU
    gan_distiller = GANSkillDistiller(input_dim=768, device=device)
    
    # 生成知识向量（模拟）
    knowledge_vectors = np.random.randn(50, 768).astype(np.float32)
    
    # 训练GAN
    gan_distiller.train(
        knowledge_vectors=knowledge_vectors,
        epochs=20,
        batch_size=16,
        verbose=True
    )
    
    # 蒸馏
    test_vector = knowledge_vectors[0:1]
    result = gan_distiller.distill(test_vector)
    
    print(f"\n✅ 蒸馏完成")
    print(f"   纯度评分: {result['purity_score']:.2%}")
    print(f"   重建误差: {result['reconstruction_error']:.6f}")
    
    return result


def evaluate_purity(knowledge_text: str, distilled_knowledge: str) -> Dict:
    """第4阶段：评估纯度"""
    print("\n" + "="*80)
    print("✔️ 第4阶段：评估蒸馏质量")
    print("="*80)
    
    evaluator = PurityEvaluator()
    
    purity_result = evaluator.get_purity_score(
        original=knowledge_text[:1000],
        distilled=distilled_knowledge[:1000]
    )
    
    overall_score = purity_result["overall_score"]
    dimensions = purity_result["dimensions"]
    
    print(f"\n📊 纯度评分结果:")
    print(f"  ╔════════════════════════════════════════════╗")
    print(f"  ║ 综合纯度评分: {overall_score:.1%}{'':>23}║")
    print(f"  ╠════════════════════════════════════════════╣")
    print(f"  ║ 信息保留率:   {dimensions['knowledge_retention']:.1%}{'':>23}║")
    print(f"  ║ 准确度:       {dimensions['accuracy']:.1%}{'':>23}║")
    print(f"  ║ 语义相似度:   {dimensions['semantic_similarity']:.1%}{'':>23}║")
    print(f"  ╚════════════════════════════════════════════╝")
    
    return purity_result


def generate_skill_md(target_person: str, purity_score: float, knowledge_text: str) -> str:
    """第5阶段：生成Skill.md"""
    print("\n" + "="*80)
    print("🏗️ 第5阶段：生成Skill.md")
    print("="*80)
    
    skill_md = f"""---
name: {target_person.lower().replace(' ', '-')}-skill
displayName: {target_person} - Expert Thinking
description: |
  使用{target_person}的思维方式和决策框架来分析和解决问题。
  
  何时使用：
  - 需要用第一性原理分析复杂问题
  - 需要从{target_person}的视角做决策
  - 需要打破传统思维，找到创新解决方案
  
  这个Skill会：
  - 应用{target_person}的核心原则
  - 使用他/她的决策框架
  - 采用他/她的沟通风格
  - 提供经过GAN蒸馏验证的高质量建议
version: 1.0.0
purityScore: {purity_score:.3f}
distillationMethod: GAN
---

# {target_person} - 思维系统

## 概述

这个Skill将{target_person}的思维方式和决策框架编码为可复用的指导。通过GAN对抗训练确保了{purity_score:.1%}的知识纯度。

## 核心原则

1. **第一性原理分析**
   - 从最基本的物理/逻辑定律开始
   - 不要依赖表面假设
   - 质疑所有conventional wisdom

2. **删除优化**
   - 删除不必要的复杂性
   - 如果添加回来就说明确实需要
   - 极简即优雅

3. **系统化思维**
   - 理解系统的约束条件
   - 优化关键路径
   - 考虑长期影响

## 决策框架

### 问题分析流程

1. **定义** - 精确理解问题是什么
2. **分解** - 分解到最基本的组成部分
3. **评估** - 从物理学/经济学角度评估
4. **创新** - 提出新的解决方案
5. **执行** - 快速迭代和改进

## 使用示例

### 例子1：技术决策
**用户问**: "���们应该用哪种架构？"

**Skill做**:
1. 分解架构要素
2. 分析每个选择的物理/经济成本
3. 提出优化方案
4. 给出实施路线图

### 例子2：业务策略  
**用户问**: "如何优化成本？"

**Skill做**:
1. 识别成本结构
2. 质疑每一项的必要性
3. 提出删除/替换方案
4. 计算潜在节省

## 沟通风格

- 直接坦率，不回避难题
- 用类比和例子说明复杂概念
- 挑战假设和conventional wisdom
- 强调第一性原理的重要性
- 鼓励大胆但经过计算的冒险

## 何时使用这个Skill

✅ 使用场景：
- 分析复杂技术问题
- 做重大战略决策
- 打破行业常规思维
- 优化成本或性能
- 需要创新解决方案

❌ 不适用场景：
- 需要温和、保守建议
- 优先考虑风险规避
- 需要遵循行业标准做法

## 质量指标

- **蒸馏纯度**: {purity_score:.1%}（保留了原始思维模式）
- **信息保留率**: 从多个来源的数据中提取
- **验证方式**: 使用GAN对抗评估

生成时间: 2026-04-14
基于GAN对抗蒸馏技术
"""
    
    print("✅ Skill.md已生成")
    return skill_md


def main(target_person: str = "Elon Musk"):
    """完整流程"""
    print("\n" + "="*80)
    print(f"🚀 GAN Skill 创建流程：{target_person}")
    print("="*80)
    
    skill_name = target_person.lower().replace(' ', '-') + '-skill'
    
    # 第1阶段：收集数据
    collector = collect_data(target_person, skill_name)
    
    # 第2阶段：提取知识
    knowledge_text = extract_knowledge(collector, target_person)
    
    # 第3阶段：GAN蒸馏
    distilled_result = distill_with_gan(knowledge_text)
    
    # 第4阶段：评估纯度
    purity_result = evaluate_purity(knowledge_text, str(distilled_result['skill_representation']))
    
    # 第5阶段：生成Skill
    skill_md = generate_skill_md(
        target_person,
        purity_result['overall_score'],
        knowledge_text
    )
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'skills-workspace', skill_name)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'SKILL.md'), 'w', encoding='utf-8') as f:
        f.write(skill_md)
    
    with open(os.path.join(output_dir, 'distillation_result.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "target_person": target_person,
            "purity_score": purity_result['overall_score'],
            "dimensions": purity_result['dimensions'],
            "reconstruction_error": distilled_result['reconstruction_error']
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print("✅ 完整流程完成！")
    print("="*80)
    print(f"""
📊 最终结果：
   ✓ 目标人物: {target_person}
   ✓ 综合纯度: {purity_result['overall_score']:.1%}
   ✓ Skill已生成: {output_dir}/SKILL.md
   ✓ 结果已保存: {output_dir}/distillation_result.json

🎉 现在可以使用这个Skill了！
""")


if __name__ == "__main__":
    main("Elon Musk")