# GAN Skill Creator

基于生成对抗网络（GAN）的高纯度知识蒸馏系统，用于创建高质量的 AI Agent Skills。

## 🎯 项目简介

GAN Skill Creator 是一个创新的工具，使用 GAN 对抗训练技术从专家知识中提取高纯度的思维模式和决策框架，生成可复用的 Claude Skills。

### 核心创新

- **GAN 对抗蒸馏**：使用生成器-判别器架构确保知识提取的高纯度
- **多维度纯度评估**：从信息保留、准确度、语义相似度三个维度评估
- **端到端流程**：从数据收集到 Skill 生成的完整自动化流程

## 📊 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    GAN Skill Creator                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ 数据收集器    │───►│ 知识提取器    │───►│ GAN 蒸馏器   │  │
│  │ (Internet)   │    │ (Claude API) │    │ (PyTorch)    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                   │         │
│                                                   ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Skill 生成   │◄───│ 纯度评估     │◄───│ 向量化器     │  │
│  │ (SKILL.md)   │    │ (Claude API) │    │ (Local)      │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r gan-skill-creator/requirements.txt
```

主要依赖：
- `torch` - GAN 训练
- `anthropic` - Claude API
- `sentence-transformers` - 本地向量化

### 2. 配置 API Key

创建 `.env` 文件：

```env
# Anthropic Claude API Key
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Claude 模型
CLAUDE_MODEL=claude-sonnet-4-20250514
```

### 3. 运行测试

```bash
# 诊断环境
python diagnose.py

# 运行完整流程
python test_pipeline.py
```

## 📁 项目结构

```
GAN-Skill/
├── .env                        # API 配置（不提交）
├── .gitignore
├── main.py                     # 主入口
├── diagnose.py                 # 环境诊断
├── test_pipeline.py            # 完整测试脚本
│
└── gan-skill-creator/
    ├── SKILL.md                # Skill 定义
    ├── README.md
    ├── requirements.txt        # Python 依赖
    │
    ├── agents/
    │   └── intent_capturer.md  # 意图捕获指南
    │
    ├── tools/
    │   ├── __init__.py
    │   ├── internet_data_collector.py  # 数据收集
    │   ├── knowledge_extractor.py      # 知识提取（Claude API）
    │   ├── gan_distiller.py            # GAN 蒸馏器
    │   ├── purity_evaluator.py         # 纯度评估（Claude API）
    │   ├── llm_client.py               # LLM 客户端封装
    │   ├── local_vectorizer.py         # 本地向量化器
    │   └── offline_purity.py           # 离线纯度评估
    │
    ├── examples/
    │   ├── elon-musk-skill-demo/       # Elon Musk 示例
    │   │   ├── README.md
    │   │   └── SKILL.md
    │   ├── full_distillation_pipeline.py
    │   └── test_distriller.py
    │
    ├── references/
    │   └── skill_template.md           # Skill 模板
    │
    └── skills-workspace/               # 生成的 Skills 输出目录
```

## 🔄 完整流程（9 阶段）

```
┌─────────────────────────────────────────────────────────────────┐
│  阶段 1: Intent Capture - 意图捕获                               │
│  ├─ 确定目标人物                                                 │
│  └─ 明确 Skill 用途和触发场景                                    │
├─────────────────────────────────────────────────────────────────┤
│  阶段 2: Data Collection - 数据收集                              │
│  ├─ 视频、文章、社交媒体、书籍等多源数据                          │
│  └─ 使用 InternetDataCollector 工具                              │
├─────────────────────────────────────────────────────────────────┤
│  阶段 3: Knowledge Extraction - 知识提取                         │
│  ├─ 使用 Claude API 提取结构化知识                               │
│  └─ 输出: 核心原则、决策框架、沟通风格等                          │
├─────────────────────────────────────────────────────────────────┤
│  阶段 4: GAN Distillation - GAN 蒸馏                             │
│  ├─ 生成器: 压缩知识表示                                         │
│  ├─ 判别器: 评估知识纯度                                         │
│  └─ 目标: 达到 90%+ 纯度                                         │
├─────────────────────────────────────────────────────────────────┤
│  阶段 5: Skill Drafting - Skill 起草                             │
│  └─ 生成 SKILL.md 文件                                           │
├─────────────────────────────────────────────────────────────────┤
│  阶段 6: Testing - 测试                                          │
│  └─ 运行测试用例验证 Skill 效果                                   │
├─────────────────────────────────────────────────────────────────┤
│  阶段 7: Evaluation - 评估                                       │
│  ├─ 信息保留率                                                   │
│  ├─ 准确度                                                       │
│  └─ 语义相似度                                                   │
├─────────────────────────────────────────────────────────────────┤
│  阶段 8: Iteration - 迭代优化                                    │
│  └─ 根据评估结果改进                                             │
├─────────────────────────────────────────────────────────────────┤
│  阶段 9: Optimization - 触发优化                                 │
│  └─ 优化 Skill 触发条件                                          │
└─────────────────────────────────────────────────────────────────┘
```

## 🧠 GAN 蒸馏原理

### 架构设计

```
原始知识向量 (768-dim)
        │
        ▼
┌───────────────────┐
│  Generator (G)    │
│  ┌─────────────┐  │
│  │ 特征提取层   │  │  识别关键特征
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │ 特征增强层   │  │  强化核心模式
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │ Skill 重建   │  │  生成 Skill 表示
│  └─────────────┘  │
└───────────────────┘
        │
        ▼
Skill 表示向量 (768-dim)
        │
        ▼
┌───────────────────┐
│ Discriminator (D) │
│  ┌─────────────┐  │
│  │ 特征比较     │  │  对比原始 vs 生成
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │ 纯度评分     │  │  输出纯度分数
│  └─────────────┘  │
└───────────────────┘
```

### 损失函数

```python
# 生成器损失
L_G = 0.6 * L_adversarial + 0.3 * L_reconstruction + 0.1 * L_feature

# 判别器损失
L_D = L_real + L_fake
```

## 📊 纯度评估维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 信息保留率 | 30% | 蒸馏后保留原始知识的比例 |
| 准确度 | 40% | 事实和逻辑的准确性 |
| 语义相似度 | 30% | 与原始语义的一致性 |

## 🛠️ 工具模块

### 1. InternetDataCollector
多源数据收集器，支持：
- 视频（YouTube 等）
- 文章（博客、新闻）
- 社交媒体（Twitter、LinkedIn）
- 书籍摘录
- 代码分析

### 2. KnowledgeExtractor
使用 Claude API 从文本中提取结构化知识：
- 核心原则
- 决策框架
- 沟通风格
- 核心价值观
- 问题解决方式
- 关键洞察

### 3. GANSkillDistiller
基于 PyTorch 的 GAN 蒸馏器：
- 生成器：SkillExpertiseGenerator
- 判别器：PurityEvaluator
- 支持自定义训练参数

### 4. PurityEvaluator
使用 Claude API 进行纯度评估：
- 三维度评估
- 可自定义权重
- 详细评估报告

### 5. LocalVectorizer
本地文本向量化：
- 基于 sentence-transformers
- 多种预设模型
- 支持批量处理

### 6. OfflinePurityEvaluator
离线纯度评估（无需 API）：
- 向量相似度计算
- 分布对齐分析
- 特征保留评估

## 📝 示例：Elon Musk Skill

查看 `gan-skill-creator/examples/elon-musk-skill-demo/` 目录。

### 提取的知识

**核心原则：**
- 第一性原理思维
- 物理学方法
- 反馈循环
- 坚持不懈

**决策框架：**
- 第一性原理分解
- "白痴指数"优化
- 删除与优化策略
- 快速迭代

**纯度评分：**
- GAN 蒸馏纯度: 44.3%
- 评估纯度: 100%

## 🔧 配置选项

### 环境变量

```env
# Anthropic API
ANTHROPIC_API_KEY=your-key
CLAUDE_MODEL=claude-sonnet-4-20250514

# 向量化模型
EMBEDDING_MODEL=all-MiniLM-L6-v2

# 设备配置
DEVICE=auto  # auto/cpu/cuda
```

### 向量化器预设

| 预设 | 模型 | 维度 | 特点 |
|------|------|------|------|
| fast | all-MiniLM-L6-v2 | 384 | 快速 |
| balanced | all-mpnet-base-v2 | 768 | 平衡 |
| quality | paraphrase-mpnet-base-v2 | 768 | 高质量 |
| multilingual | paraphrase-multilingual-mpnet-base-v2 | 768 | 多语言 |
| chinese | shibing624/text2vec-base-chinese | 768 | 中文优化 |

## 📚 API 参考

### create_skill()

```python
from main import create_skill

result = create_skill(
    target_person="Elon Musk",
    skill_name="elon-musk-skill",  # 可选
    verbose=True
)

# 返回值
# {
#     'skill_md': '...',           # Skill.md 内容
#     'purity_score': 0.85,        # 纯度评分
#     'output_dir': '...',         # 输出目录
#     'knowledge': {...},          # 提取的知识
#     'distilled': {...}           # 蒸馏结果
# }
```

### GANSkillDistiller

```python
from tools.gan_distiller import GANSkillDistiller

# 初始化
distiller = GANSkillDistiller(
    input_dim=384,  # 向量维度
    device='cpu'    # 设备
)

# 训练
distiller.train(
    knowledge_vectors=vectors,  # (N, D) 数组
    epochs=50,
    batch_size=16,
    verbose=True
)

# 蒸馏
result = distiller.distill(knowledge_vector)
# result['purity_score']  # 纯度评分
```

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 许可证

MIT License

## 🙏 致谢

- [Anthropic](https://www.anthropic.com/) - Claude API
- [Sentence Transformers](https://www.sbert.net/) - 文本向量化
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

**Created with ❤️ by GAN Skill Creator**
