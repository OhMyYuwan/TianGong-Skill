"""
离线纯度评估：不依赖 LLM API 的纯度评估方法
使用向量相似度和统计方法
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 尝试导入向量化器
try:
    from .local_vectorizer import LocalVectorizer, get_vectorizer
    HAS_VECTORIZER = True
except ImportError:
    HAS_VECTORIZER = False


@dataclass
class PurityResult:
    """纯度评估结果"""
    overall_score: float
    knowledge_retention: float
    semantic_similarity: float
    distribution_alignment: float
    feature_preservation: float
    details: Dict


class OfflinePurityEvaluator:
    """
    离线纯度评估器

    不依赖 LLM API，使用向量计算和统计方法评估：
    1. 知识保留率 - 通过信息熵和关键特征保留
    2. 语义相似度 - 通过向量余弦相似度
    3. 分布对齐 - 通过 KL 散度或 Wasserstein 距离
    4. 特征保留 - 通过 PCA/特征值分析
    """

    def __init__(
        self,
        vectorizer: Optional[LocalVectorizer] = None,
        vectorizer_preset: str = "fast"
    ):
        """
        初始化评估器

        Args:
            vectorizer: 向量化器实例
            vectorizer_preset: 如果未提供 vectorizer，使用此预设创建
        """
        if vectorizer is not None:
            self.vectorizer = vectorizer
        elif HAS_VECTORIZER:
            self.vectorizer = get_vectorizer(preset=vectorizer_preset)
        else:
            self.vectorizer = None
            print("⚠️ 向量化器不可用，部分功能受限")

    def evaluate_knowledge_retention(
        self,
        original: np.ndarray,
        distilled: np.ndarray
    ) -> float:
        """
        评估知识保留率

        使用向量重建误差和信息保留度量

        Args:
            original: 原始知识向量
            distilled: 蒸馏后向量

        Returns:
            保留率 [0, 1]
        """
        # 方法1: 余弦相似度
        cos_sim = np.dot(original, distilled) / (
            np.linalg.norm(original) * np.linalg.norm(distilled) + 1e-8
        )

        # 方法2: 相对重建误差
        reconstruction_error = np.mean((original - distilled) ** 2)
        original_energy = np.mean(original ** 2) + 1e-8
        relative_error = reconstruction_error / original_energy
        retention_from_error = max(0, 1 - relative_error)

        # 综合评分
        retention = 0.6 * cos_sim + 0.4 * retention_from_error

        return float(np.clip(retention, 0, 1))

    def evaluate_semantic_similarity(
        self,
        original_text: str,
        distilled_text: str
    ) -> float:
        """
        评估语义相似度

        Args:
            original_text: 原始文本
            distilled_text: 蒸馏后文本

        Returns:
            相似度 [0, 1]
        """
        if self.vectorizer is None:
            # 回退到简单的词汇重叠
            orig_words = set(original_text.lower().split())
            dist_words = set(distilled_text.lower().split())

            if not orig_words:
                return 0.0

            overlap = len(orig_words & dist_words) / len(orig_words)
            return overlap

        # 使用向量化器计算语义相似度
        orig_vec = self.vectorizer.vectorize(original_text)
        dist_vec = self.vectorizer.vectorize(distilled_text)

        similarity = self.vectorizer.similarity(orig_vec, dist_vec)

        # 将 [-1, 1] 映射到 [0, 1]
        return float((similarity + 1) / 2)

    def evaluate_distribution_alignment(
        self,
        original_vectors: np.ndarray,
        distilled_vectors: np.ndarray
    ) -> float:
        """
        评估分布对齐度

        使用统计矩匹配（均值、方差、偏度）

        Args:
            original_vectors: 原始向量集合 [N, D]
            distilled_vectors: 蒸馏向量集合 [M, D]

        Returns:
            对齐度 [0, 1]
        """
        # 计算统计矩
        def compute_moments(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            mean = np.mean(vecs, axis=0)
            var = np.var(vecs, axis=0)
            # 偏度
            std = np.sqrt(var + 1e-8)
            skew = np.mean(((vecs - mean) / std) ** 3, axis=0)
            return mean, var, skew

        orig_mean, orig_var, orig_skew = compute_moments(original_vectors)
        dist_mean, dist_var, dist_skew = compute_moments(distilled_vectors)

        # 计算矩匹配分数
        mean_match = 1 - np.mean(np.abs(orig_mean - dist_mean)) / (np.mean(np.abs(orig_mean)) + 1e-8)
        var_match = 1 - np.mean(np.abs(orig_var - dist_var)) / (np.mean(np.abs(orig_var)) + 1e-8)
        skew_match = 1 - np.mean(np.abs(orig_skew - dist_skew)) / (np.mean(np.abs(orig_skew)) + 1e-8)

        # 综合评分
        alignment = 0.4 * mean_match + 0.4 * var_match + 0.2 * skew_match

        return float(np.clip(alignment, 0, 1))

    def evaluate_feature_preservation(
        self,
        original: np.ndarray,
        distilled: np.ndarray,
        top_k_features: int = 50
    ) -> float:
        """
        评估关键特征保留

        通过分析最大幅值特征的保留情况

        Args:
            original: 原始向量
            distilled: 蒸馏向量
            top_k_features: 考虑前 k 个最大特征

        Returns:
            特征保留率 [0, 1]
        """
        # 找到原始向量中幅值最大的特征索引
        abs_original = np.abs(original)
        top_indices = np.argsort(abs_original)[-top_k_features:]

        # 计算这些特征在蒸馏向量中的保留情况
        orig_top_values = original[top_indices]
        dist_top_values = distilled[top_indices]

        # 符号一致性
        sign_match = np.mean(np.sign(orig_top_values) == np.sign(dist_top_values))

        # 相对幅值保留
        orig_magnitudes = np.abs(orig_top_values)
        dist_magnitudes = np.abs(dist_top_values)
        magnitude_retention = np.mean(np.minimum(dist_magnitudes, orig_magnitudes) / (orig_magnitudes + 1e-8))

        # 综合评分
        preservation = 0.5 * sign_match + 0.5 * magnitude_retention

        return float(preservation)

    def evaluate(
        self,
        original: np.ndarray,
        distilled: np.ndarray,
        original_text: Optional[str] = None,
        distilled_text: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> PurityResult:
        """
        综合评估

        Args:
            original: 原始向量
            distilled: 蒸馏向量
            original_text: 原始文本（可选，用于语义相似度）
            distilled_text: 蒸馏文本（可选）
            weights: 各维度的权重

        Returns:
            PurityResult 对象
        """
        if weights is None:
            weights = {
                "retention": 0.35,
                "similarity": 0.25,
                "distribution": 0.20,
                "feature": 0.20
            }

        # 确保是 numpy 数组
        original = np.asarray(original)
        distilled = np.asarray(distilled)

        # 处理批次维度
        if original.ndim == 1:
            original = original.reshape(1, -1)
        if distilled.ndim == 1:
            distilled = distilled.reshape(1, -1)

        # 计算各维度分数
        knowledge_retention = self.evaluate_knowledge_retention(
            original.mean(axis=0), distilled.mean(axis=0)
        )

        if original_text and distilled_text:
            semantic_similarity = self.evaluate_semantic_similarity(
                original_text, distilled_text
            )
        else:
            # 使用向量相似度代替
            semantic_similarity = float(
                np.dot(original.mean(axis=0), distilled.mean(axis=0)) /
                (np.linalg.norm(original.mean(axis=0)) * np.linalg.norm(distilled.mean(axis=0)) + 1e-8)
            )
            semantic_similarity = (semantic_similarity + 1) / 2  # 映射到 [0, 1]

        distribution_alignment = self.evaluate_distribution_alignment(original, distilled)

        feature_preservation = self.evaluate_feature_preservation(
            original.mean(axis=0), distilled.mean(axis=0)
        )

        # 计算综合分数
        overall_score = (
            weights["retention"] * knowledge_retention +
            weights["similarity"] * semantic_similarity +
            weights["distribution"] * distribution_alignment +
            weights["feature"] * feature_preservation
        )

        return PurityResult(
            overall_score=overall_score,
            knowledge_retention=knowledge_retention,
            semantic_similarity=semantic_similarity,
            distribution_alignment=distribution_alignment,
            feature_preservation=feature_preservation,
            details={
                "weights": weights,
                "num_samples_original": original.shape[0],
                "num_samples_distilled": distilled.shape[0],
                "vector_dim": original.shape[1]
            }
        )

    def print_result(self, result: PurityResult):
        """打印评估结果"""
        print("\n" + "╔" + "═" * 50 + "╗")
        print(f"║{'离线纯度评估结果':^48}║")
        print("╠" + "═" * 50 + "╣")
        print(f"║ 综合纯度评分: {result.overall_score:>8.1%}{' ' * 26}║")
        print("╠" + "─" * 50 + "╣")
        print(f"║ 知识保留率:   {result.knowledge_retention:>8.1%}{' ' * 26}║")
        print(f"║ 语义相似度:   {result.semantic_similarity:>8.1%}{' ' * 26}║")
        print(f"║ 分布对齐度:   {result.distribution_alignment:>8.1%}{' ' * 26}║")
        print(f"║ 特征保留率:   {result.feature_preservation:>8.1%}{' ' * 26}║")
        print("╚" + "═" * 50 + "╝")


if __name__ == "__main__":
    # 测试
    print("=== 测试 OfflinePurityEvaluator ===\n")

    evaluator = OfflinePurityEvaluator()

    # 模拟数据
    np.random.seed(42)
    original = np.random.randn(10, 768).astype(np.float32)

    # 模拟蒸馏（添加少量噪声）
    distilled = original + 0.1 * np.random.randn(10, 768).astype(np.float32)

    # 评估
    result = evaluator.evaluate(original, distilled)
    evaluator.print_result(result)

    # 测试文本语义相似度
    print("\n测试语义相似度:")
    sim = evaluator.evaluate_semantic_similarity(
        "第一性原理是从基本物理定律出发分析问题",
        "第一性原理思维要求从物理学基础推导解决方案"
    )
    print(f"相似度: {sim:.2%}")
