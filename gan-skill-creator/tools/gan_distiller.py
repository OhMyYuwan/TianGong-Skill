"""
GAN-based Knowledge Distillation Engine
真正的生成对抗网络，用于高纯度知识蒸馏
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional


class SkillExpertiseGenerator(nn.Module):
    """
    生成器：从原始知识生成"Skill精粹"
    
    目标：保留95%+的关键特征，同时压缩到可用大小
    """
    def __init__(self, input_dim=768, latent_dim=256):
        super().__init__()
        
        # 特征提取：识别关键特征
        self.key_feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        
        # 特征增强：强化关键特征
        self.feature_enhancer = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # 重建：输出完整Skill表示
        self.skill_reconstructor = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 原始知识 (batch, 768)
        Returns:
            skill_output: Skill表示 (batch, 768)
            key_features: 关键特征 (batch, latent_dim)
            reconstruction: 重建 (batch, 768)
        """
        key_features = self.key_feature_extractor(x)
        enhanced_features = self.feature_enhancer(key_features)
        skill_output = self.skill_reconstructor(enhanced_features)
        reconstruction = skill_output
        
        return skill_output, key_features, reconstruction


class PurityEvaluator(nn.Module):
    """
    判别器：评估Skill的"纯度"
    
    纯度定义：
    - 保留了原始知识的核心特征吗？(90%+)
    - 没有引入虚假特征吗？
    - 是否适合作为Skill？
    """
    def __init__(self, input_dim=768):
        super().__init__()
        
        # 特征比较：对比原始vs生成
        self.comparator = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 纯度评分
        self.purity_scorer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, original: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        """
        Args:
            original: 原始知识 (batch, 768)
            generated: 生成的Skill (batch, 768)
        Returns:
            purity_score: 纯度评��� (batch, 1) [0, 1]
        """
        combined = torch.cat([original, generated], dim=1)
        features = self.comparator(combined)
        purity = self.purity_scorer(features)
        return purity


class GANSkillDistiller:
    """
    完整的GAN蒸馏系统，特别优化用于生成高质量Skill
    """
    
    def __init__(self, input_dim=768, device='cpu'):
        self.device = device
        self.input_dim = input_dim
        
        self.generator = SkillExpertiseGenerator(input_dim).to(device)
        self.discriminator = PurityEvaluator(input_dim).to(device)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        self.training_history = {
            "epochs": [],
            "g_loss": [],
            "d_loss": [],
            "purity": []
        }
    
    def train(self, knowledge_vectors: np.ndarray, epochs=50, batch_size=32, verbose=True):
        """
        训练GAN
        
        Args:
            knowledge_vectors: (num_samples, 768) 知识向量
            epochs: 训练轮数
            batch_size: 批大小
            verbose: 是否打印进度
        """
        if not isinstance(knowledge_vectors, torch.Tensor):
            knowledge_vectors = torch.tensor(knowledge_vectors, dtype=torch.float32)
        
        knowledge_vectors = knowledge_vectors.to(self.device)
        num_samples = knowledge_vectors.shape[0]
        
        if verbose:
            print("🚀 GAN训练开始...")
            print(f"   数据: {num_samples} 样本")
            print(f"   Epochs: {epochs}")
            print("="*70)
        
        for epoch in range(epochs):
            # 随机打乱
            indices = torch.randperm(num_samples).to(self.device)
            
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_purity = 0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_data = knowledge_vectors[batch_indices]
                
                # ===== 训练判别器 =====
                self.d_optimizer.zero_grad()
                
                # 生成器生成Skill
                generated_skill, _, _ = self.generator(batch_data)
                
                # 判别器评估
                real_purity = self.discriminator(batch_data, batch_data)
                fake_purity = self.discriminator(batch_data, generated_skill.detach())
                
                # 判别器损失
                d_loss_real = self.bce_loss(real_purity, torch.ones_like(real_purity))
                d_loss_fake = self.bce_loss(fake_purity, torch.zeros_like(fake_purity))
                d_loss = d_loss_real + d_loss_fake
                
                d_loss.backward()
                self.d_optimizer.step()
                
                # ===== 训练生成器 =====
                self.g_optimizer.zero_grad()
                
                generated_skill, key_features, reconstruction = self.generator(batch_data)
                fake_purity = self.discriminator(batch_data, generated_skill)
                
                # 生成器损失：3个部分
                g_loss_adversarial = self.bce_loss(fake_purity, torch.ones_like(fake_purity))
                g_loss_reconstruction = self.mse_loss(reconstruction, batch_data)
                g_loss_feature = self.mse_loss(key_features, key_features)
                
                g_loss = 0.6 * g_loss_adversarial + 0.3 * g_loss_reconstruction + 0.1 * g_loss_feature
                
                g_loss.backward()
                self.g_optimizer.step()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_purity += fake_purity.mean().item()
                num_batches += 1
            
            # 平均损失
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            avg_purity = epoch_purity / num_batches
            
            self.training_history["epochs"].append(epoch + 1)
            self.training_history["g_loss"].append(avg_g_loss)
            self.training_history["d_loss"].append(avg_d_loss)
            self.training_history["purity"].append(avg_purity)
            
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"G_Loss: {avg_g_loss:7.4f} | "
                    f"D_Loss: {avg_d_loss:7.4f} | "
                    f"Purity: {avg_purity:6.1%}"
                )
        
        if verbose:
            print("="*70)
            print(f"✅ 训练完成！最终纯度: {avg_purity:.1%}")
    
    def distill(self, knowledge_vector: np.ndarray) -> Dict:
        """
        执行单次蒸馏
        """
        with torch.no_grad():
            if isinstance(knowledge_vector, np.ndarray):
                knowledge_vector = torch.tensor(knowledge_vector, dtype=torch.float32)
            
            if knowledge_vector.dim() == 1:
                knowledge_vector = knowledge_vector.unsqueeze(0)
            
            knowledge_vector = knowledge_vector.to(self.device)
            
            skill_output, key_features, reconstruction = self.generator(knowledge_vector)
            purity = self.discriminator(knowledge_vector, skill_output)
        
        return {
            "skill_representation": skill_output.cpu().numpy(),
            "key_features": key_features.cpu().numpy(),
            "purity_score": purity.cpu().item(),
            "reconstruction_error": np.mean(
                (knowledge_vector.cpu().numpy() - skill_output.cpu().numpy()) ** 2
            )
        }
    
    def get_training_history(self) -> Dict:
        """获取训练历史"""
        return self.training_history