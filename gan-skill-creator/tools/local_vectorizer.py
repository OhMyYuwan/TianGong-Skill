"""
本地向量化器：使用本地模型进行文本向量化
支持多种 embedding 后端
"""
from typing import List, Union, Optional
import numpy as np

# 尝试导入多种 embedding 后端
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class LocalVectorizer:
    """
    本地向量化器

    支持的后端：
    - sentence-transformers: 推荐使用，高效且质量好
    - transformers: 直接使用 HuggingFace 模型
    - random: 随机向量（仅用于测试）
    """

    def __init__(
        self,
        backend: str = "sentence-transformers",
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        初始化向量化器

        Args:
            backend: 后端类型
            model_name: 模型名称
            device: 设备 (cuda/cpu/auto)
        """
        self.backend = backend
        self.model_name = model_name

        # 确定设备
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 初始化模型
        if backend == "sentence-transformers":
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers 未安装。"
                    "请运行: pip install sentence-transformers"
                )
            print(f"📦 加载 SentenceTransformer 模型: {model_name}")
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

        elif backend == "transformers":
            if not HAS_TRANSFORMERS:
                raise ImportError(
                    "transformers 未安装。"
                    "请运行: pip install transformers torch"
                )
            print(f"📦 加载 Transformers 模型: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.embedding_dim = self.model.config.hidden_size

        elif backend == "random":
            print("⚠️ 使用随机向量（仅用于测试）")
            self.model = None
            self.embedding_dim = 768

        else:
            raise ValueError(f"不支持的后端: {backend}")

        print(f"   设备: {self.device}")
        print(f"   向量维度: {self.embedding_dim}")

    def vectorize(self, text: str) -> np.ndarray:
        """
        将文本转换为向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量 (numpy array)
        """
        if self.backend == "sentence-transformers":
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding

        elif self.backend == "transformers":
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                outputs = self.model(**inputs)
                # 使用 [CLS] token 的输出作为句子表示
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                return embedding.squeeze()

        elif self.backend == "random":
            return np.random.randn(self.embedding_dim).astype(np.float32)

    def vectorize_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量向量化（更高效）

        Args:
            texts: 文本列表
            batch_size: 批大小

        Returns:
            嵌入向量矩阵 (numpy array, shape: [num_texts, embedding_dim])
        """
        if self.backend == "sentence-transformers":
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100
            )
            return embeddings

        elif self.backend == "transformers":
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                with torch.no_grad():
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    ).to(self.device)
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(batch_embeddings)
            return np.vstack(embeddings)

        elif self.backend == "random":
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            相似度分数 [-1, 1]
        """
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def find_similar(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 5
    ) -> List[dict]:
        """
        在语料库中查找相似文本

        Args:
            query: 查询文本
            corpus: 语料库
            top_k: 返回前 k 个结果

        Returns:
            相似结果列表，每个元素包含 text, similarity, index
        """
        query_vec = self.vectorize(query)
        corpus_vecs = self.vectorize_batch(corpus)

        # 计算相似度
        similarities = [
            {
                "text": corpus[i],
                "similarity": self.similarity(query_vec, corpus_vecs[i]),
                "index": i
            }
            for i in range(len(corpus))
        ]

        # 排序并返回 top_k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]


# 预设模型配置
MODEL_PRESETS = {
    "fast": {
        "backend": "sentence-transformers",
        "model_name": "all-MiniLM-L6-v2",  # 384 维，快速
    },
    "balanced": {
        "backend": "sentence-transformers",
        "model_name": "all-mpnet-base-v2",  # 768 维，平衡
    },
    "quality": {
        "backend": "sentence-transformers",
        "model_name": "paraphrase-mpnet-base-v2",  # 768 维，高质量
    },
    "multilingual": {
        "backend": "sentence-transformers",
        "model_name": "paraphrase-multilingual-mpnet-base-v2",  # 支持多语言
    },
    "chinese": {
        "backend": "sentence-transformers",
        "model_name": "shibing624/text2vec-base-chinese",  # 中文优化
    }
}


def get_vectorizer(preset: str = "fast", **kwargs) -> LocalVectorizer:
    """
    使用预设配置获取向量化器

    Args:
        preset: 预设名称 (fast/balanced/quality/multilingual/chinese)
        **kwargs: 其他参数覆盖预设

    Returns:
        LocalVectorizer 实例
    """
    if preset not in MODEL_PRESETS:
        raise ValueError(f"未知预设: {preset}。可用: {list(MODEL_PRESETS.keys())}")

    config = MODEL_PRESETS[preset].copy()
    config.update(kwargs)

    return LocalVectorizer(**config)


if __name__ == "__main__":
    # 测试
    print("=== 测试 LocalVectorizer ===\n")

    vectorizer = LocalVectorizer(backend="sentence-transformers")

    # 单文本测试
    text = "这是一个测试句子。"
    vec = vectorizer.vectorize(text)
    print(f"文本: {text}")
    print(f"向量形状: {vec.shape}")
    print(f"向量前5维: {vec[:5]}\n")

    # 批量测试
    texts = [
        "第一性原理是一种思维方式",
        "从基本物理定律出发分析问题",
        "今天天气真不错"
    ]
    vecs = vectorizer.vectorize_batch(texts)
    print(f"批量向量形状: {vecs.shape}\n")

    # 相似度测试
    sim = vectorizer.similarity(vecs[0], vecs[1])
    print(f"'{texts[0]}' vs '{texts[1]}' 相似度: {sim:.4f}")

    sim = vectorizer.similarity(vecs[0], vecs[2])
    print(f"'{texts[0]}' vs '{texts[2]}' 相似度: {sim:.4f}\n")

    # 预设测试
    print("可用预设:", list(MODEL_PRESETS.keys()))
