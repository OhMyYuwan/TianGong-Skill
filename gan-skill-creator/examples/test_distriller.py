"""
测试脚本：验证GAN蒸馏器的功能
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tools.gan_distiller import GANSkillDistiller


def test_basic_distillation():
    """测试基础蒸馏功能"""
    print("\n" + "="*80)
    print("🧪 测试1: 基础蒸馏")
    print("="*80)
    
    distiller = GANSkillDistiller(input_dim=768, device='cpu')
    
    # 生成随机知识向量
    knowledge = np.random.randn(1, 768).astype(np.float32)
    
    # 蒸馏
    result = distiller.distill(knowledge)
    
    print(f"✅ 通过!")
    print(f"   纯度评分: {result['purity_score']:.3f}")
    print(f"   重建误差: {result['reconstruction_error']:.6f}")
    
    assert 0 <= result['purity_score'] <= 1, "纯度评分应在0-1之间"
    assert result['reconstruction_error'] >= 0, "重建误差应为正数"


def test_generator_reconstruction():
    """测试生成器重建"""
    print("\n" + "="*80)
    print("🧪 测试2: 生成器重建")
    print("="*80)
    
    distiller = GANSkillDistiller(input_dim=768, device='cpu')
    x = np.random.randn(4, 768).astype(np.float32)
    
    import torch
    x_tensor = torch.tensor(x, dtype=torch.float32)
    skill_output, latent, reconstruction = distiller.generator(x_tensor)
    
    print(f"✅ 通过!")
    print(f"   输入形状: {x_tensor.shape}")
    print(f"   Skill输出形状: {skill_output.shape}")
    print(f"   隐层形状: {latent.shape}")
    
    assert skill_output.shape == x_tensor.shape, "输出形状应与输入相同"
    assert latent.shape[1] == 256, "隐层维度应为256"


def test_discriminator():
    """测试判别器"""
    print("\n" + "="*80)
    print("🧪 测试3: 判别器评估")
    print("="*80)
    
    distiller = GANSkillDistiller(input_dim=768, device='cpu')
    
    import torch
    original = torch.randn(1, 768)
    generated = torch.randn(1, 768)
    
    purity = distiller.discriminator(original, generated)
    
    print(f"✅ 通过!")
    print(f"   纯度评分: {purity.item():.3f}")
    
    assert 0 <= purity.item() <= 1, "纯度评分应在0-1之间"


def test_training():
    """测试GAN训练"""
    print("\n" + "="*80)
    print("🧪 测试4: GAN训练（简短版本）")
    print("="*80)
    
    distiller = GANSkillDistiller(input_dim=768, device='cpu')
    
    # 生成小规模训练数据
    knowledge_vectors = np.random.randn(20, 768).astype(np.float32)
    
    # 训练（只用5个epoch进行快速测试）
    distiller.train(
        teacher_outputs=knowledge_vectors,
        epochs=5,
        batch_size=8,
        verbose=False
    )
    
    print(f"✅ 通过!")
    print(f"   训练历史:")
    print(f"   - 训练轮数: {len(distiller.training_history['epochs'])}")
    print(f"   - 最终纯度: {distiller.training_history['purity'][-1]:.1%}")
    
    assert len(distiller.training_history['epochs']) == 5, "应该有5个epoch的记录"


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 GAN Skill Distiller 完整测试套件")
    print("="*80)
    
    try:
        test_basic_distillation()
        test_generator_reconstruction()
        test_discriminator()
        test_training()
        
        print("\n" + "="*80)
        print("✅ 所有测试通过！")
        print("="*80)
        return True
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)