"""
知识提取工具：从互联网资料中提取结构化知识
"""
from anthropic import Anthropic
from typing import Dict, List, Optional
import json

# 尝试导入 sentence-transformers，如果不可用则使用模拟
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    import numpy as np


class KnowledgeExtractor:
    """使用Claude API从文本中提取知识"""

    def __init__(self, api_key: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = Anthropic(api_key=api_key)

        # 初始化 embedding 模型
        self.embedding_model_name = embedding_model
        if HAS_SENTENCE_TRANSFORMERS:
            print(f"📦 加载 embedding 模型: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"   向量维度: {self.embedding_dim}")
        else:
            print("⚠️ sentence-transformers 未安装，使用模拟 embedding")
            self.embedding_model = None
            self.embedding_dim = 768
    
    def extract_from_sources(self, target_person: str, combined_text: str) -> Dict:
        """
        从多源资料中提取知识
        """
        print(f"\n🧠 使用Claude提取 {target_person} 的知识...")
        
        extraction_prompt = f"""
你是知识提取专家。请从以下关于{target_person}的资料中提取结构化知识。

资料：
{combined_text[:3000]}... （内容已截断以保持简洁）

请提取以下维度的知识，返回JSON格式：

{{
    "core_principles": [
        "原则1及其解释",
        "原则2及其解释"
    ],
    "decision_frameworks": [
        "决策框架1的步骤",
        "决策框架2的步骤"
    ],
    "communication_style": "描述他/她的沟通方式",
    "core_values": ["值1", "值2", "值3"],
    "problem_solving_approach": "解决问题的方式",
    "key_insights": [
        "洞察1",
        "洞察2"
    ]
}}

只返回JSON，不要其他文字。
"""
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": extraction_prompt
            }]
        )
        
        response_text = response.content[0].text
        
        try:
            knowledge = json.loads(response_text)
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回文本
            knowledge = {
                "raw_extraction": response_text,
                "error": "JSON解析失败，返回原始文本"
            }
        
        print("✅ 知识提取完成")
        return knowledge
    
    def vectorize_knowledge(self, knowledge: Dict) -> str:
        """
        将结构化知识转换为文本表示（稍后会被转为向量）
        """
        vectorized = f"""
核心原则：
{json.dumps(knowledge.get('core_principles', []), ensure_ascii=False, indent=2)}

决策框架：
{json.dumps(knowledge.get('decision_frameworks', []), ensure_ascii=False, indent=2)}

沟通风格：
{knowledge.get('communication_style', 'N/A')}

核心价值观：
{json.dumps(knowledge.get('core_values', []), ensure_ascii=False, indent=2)}

问题解决方式：
{knowledge.get('problem_solving_approach', 'N/A')}

关键洞察：
{json.dumps(knowledge.get('key_insights', []), ensure_ascii=False, indent=2)}
"""
        return vectorized
    
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量

        Args:
            text: 要向量化的文本

        Returns:
            嵌入向量 (list of floats)
        """
        if self.embedding_model is not None:
            # 使用真实的 sentence-transformers 模型
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            # 回退到模拟向量
            import numpy as np
            embedding = np.random.randn(self.embedding_dim).tolist()
            return embedding

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取嵌入向量（更高效）

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        if self.embedding_model is not None:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            import numpy as np
            return [np.random.randn(self.embedding_dim).tolist() for _ in texts]


if __name__ == "__main__":
    import os
    
    extractor = KnowledgeExtractor()
    
    # 示例使用
    sample_text = """
    Elon Musk的思想：
    - 第一性原理分析
    - 从物理学基础推导
    - 删除不必要的部分
    """
    
    knowledge = extractor.extract_from_sources("Elon Musk", sample_text)
    print("提取的知识：")
    print(json.dumps(knowledge, ensure_ascii=False, indent=2))