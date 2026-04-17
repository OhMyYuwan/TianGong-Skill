"""
诊断脚本 - 检查环境和依赖
"""
import sys
import os
from pathlib import Path

print("=" * 60)
print("🔍 GAN Skill Creator 环境诊断")
print("=" * 60)

# 1. Python 版本
print(f"\n📌 Python 版本: {sys.version}")
print(f"   Python 路径: {sys.executable}")

# 2. 工作目录
print(f"\n📌 工作目录: {os.getcwd()}")
print(f"   脚本位置: {Path(__file__).parent}")

# 3. 环境变量
print("\n📌 环境变量:")
key = os.environ.get('ANTHROPIC_API_KEY', '')
print(f"   ANTHROPIC_API_KEY: {key[:20] if key else '❌ 未设置'}...")

# 4. 加载 .env
print("\n📌 尝试加载 .env 文件:")
env_path = Path(__file__).parent / '.env'
print(f"   .env 路径: {env_path}")
print(f"   .env 存在: {env_path.exists()}")

if env_path.exists():
    try:
        with open(env_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip())
        print("   ✅ .env 手动加载成功")
        key = os.environ.get('ANTHROPIC_API_KEY', '')
        if key and key != "your-anthropic-api-key-here":
            print(f"   ANTHROPIC_API_KEY: {key[:20]}... (长度: {len(key)})")
        else:
            print("   ⚠️ ANTHROPIC_API_KEY 未配置")
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")

# 5. 检查依赖
print("\n📌 检查依赖包:")
dependencies = [
    'torch',
    'numpy',
    'anthropic',
    'sentence_transformers',
    'dotenv'
]

for dep in dependencies:
    try:
        if dep == 'dotenv':
            import dotenv
            print(f"   ✅ python-dotenv")
        elif dep == 'sentence_transformers':
            import sentence_transformers
            print(f"   ✅ sentence-transformers")
        else:
            __import__(dep)
            print(f"   ✅ {dep}")
    except ImportError as e:
        print(f"   ❌ {dep}: {e}")

# 6. 测试导入项目模块
print("\n📌 测试导入项目模块:")
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'gan-skill-creator'))

modules = [
    ('tools.internet_data_collector', 'InternetDataCollector'),
    ('tools.knowledge_extractor', 'KnowledgeExtractor'),
    ('tools.gan_distiller', 'GANSkillDistiller'),
    ('tools.purity_evaluator', 'PurityEvaluator'),
    ('tools.local_vectorizer', 'get_vectorizer'),
]

for module_name, attr in modules:
    try:
        module = __import__(module_name, fromlist=[attr])
        getattr(module, attr)
        print(f"   ✅ {module_name}.{attr}")
    except Exception as e:
        print(f"   ❌ {module_name}.{attr}: {e}")

# 7. 测试 Claude API 连接
print("\n📌 测试 Claude API 连接:")
api_key = os.environ.get('ANTHROPIC_API_KEY')
if api_key and api_key != "your-anthropic-api-key-here":
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{"role": "user", "content": "请回复'测试成功'"}]
        )
        print(f"   ✅ Claude API 连接成功")
        print(f"   响应: {response.content[0].text[:50]}")
    except Exception as e:
        print(f"   ❌ Claude API 连接失败: {e}")
else:
    print("   ⚠️ 跳过（API Key 未配置）")

print("\n" + "=" * 60)
print("✅ 诊断完成")
print("=" * 60)
