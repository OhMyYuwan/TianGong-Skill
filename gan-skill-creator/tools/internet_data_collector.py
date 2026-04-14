"""
互联网数据收集器：从网络上收集人物资料
"""
import json
import os
from datetime import datetime
from typing import List, Dict


class InternetDataCollector:
    """从互联网收集关于某个人的数据"""
    
    def __init__(self, target_person: str, skill_name: str):
        self.target_person = target_person
        self.skill_name = skill_name
        self.data_sources = {
            "videos": [],
            "articles": [],
            "code": [],
            "interviews": [],
            "social_media": [],
            "books": [],
            "academic": []
        }
        self.collected_at = datetime.now().isoformat()
    
    def print_collection_guide(self):
        """打印数据收集指南"""
        guide = f"""
{'='*80}
📚 数据收集指南：{self.target_person}
{'='*80}

📍 视频（Videos）
  来源：
    • YouTube - Search "{self.target_person} interview"
    • TED - https://www.ted.com/search
    • Podcast - Various platforms
  步骤：
    1. 找有自动字幕的视频
    2. 右键 → Show transcript
    3. 复制全部文本
    4. 粘贴到这里

📍 文章（Articles）
  来源：
    • Medium - Search for articles
    • 个人博客/网站
    • 新闻文章
    • Wikipedia
  步骤：
    1. 复制相关文本
    2. 包含标题、作者、日期
    3. 粘贴到这里

📍 社交媒体（Social Media）
  来源：
    • Twitter/X - @username
    • LinkedIn - 个人资料
    • Instagram - 帖子
  步骤：
    1. 找代表性帖子
    2. 复制文本（不需要图片）
    3. 添加5-10条关键帖子

📍 书籍（Books）
  来源：
    • Amazon - Search by author
    • Goodreads - 引用和评论
    • Google Books - 预览
  步骤：
    1. 找自传或关键哲学书籍
    2. 复制关键段落（2-3个/书籍）
    3. 记录标题和页码

📍 代码（Code - if applicable）
  来源：
    • GitHub - 个人仓库
    • 开源项目
    • 代码示例
  步骤：
    1. 访问GitHub
    2. 查看README和代码注释
    3. 分析编码风格
    4. 记录模式

{'='*80}
💡 提示：
  • 质量 > 数量 （3个好采访 > 10条推文）
  • 不需要收集所有类别
  • 保存每条数据的URL来源
{'='*80}
"""
        print(guide)
    
    def add_video(self, title: str, transcript: str, source_url: str = ""):
        """添加视频"""
        self.data_sources["videos"].append({
            "title": title,
            "transcript": transcript,
            "source_url": source_url,
            "added_at": datetime.now().isoformat()
        })
        return len(self.data_sources["videos"])
    
    def add_article(self, title: str, content: str, source_url: str = ""):
        """添加文章"""
        self.data_sources["articles"].append({
            "title": title,
            "content": content,
            "source_url": source_url,
            "added_at": datetime.now().isoformat()
        })
        return len(self.data_sources["articles"])
    
    def add_post(self, platform: str, text: str, post_url: str = ""):
        """添加社交媒体帖子"""
        self.data_sources["social_media"].append({
            "platform": platform,
            "text": text,
            "post_url": post_url,
            "added_at": datetime.now().isoformat()
        })
        return len(self.data_sources["social_media"])
    
    def add_book(self, book_title: str, excerpts: List[str], author: str = ""):
        """添加书籍摘录"""
        self.data_sources["books"].append({
            "book_title": book_title,
            "author": author,
            "excerpts": excerpts,
            "added_at": datetime.now().isoformat()
        })
        return len(self.data_sources["books"])
    
    def add_code_analysis(self, repo_name: str, analysis: str):
        """添加代码分析"""
        self.data_sources["code"].append({
            "repo": repo_name,
            "analysis": analysis,
            "added_at": datetime.now().isoformat()
        })
        return len(self.data_sources["code"])
    
    def get_statistics(self) -> Dict:
        """获取收集统计"""
        stats = {"target": self.target_person, "sources": {}}
        for source_type, items in self.data_sources.items():
            if items:
                stats["sources"][source_type] = len(items)
        stats["total_items"] = sum(len(items) for items in self.data_sources.values())
        return stats
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print(f"\n📊 收集统计：{stats['target']}")
        print(f"{'='*50}")
        for source_type, count in stats["sources"].items():
            print(f"  {source_type:15s}: {count:3d} 项")
        print(f"{'='*50}")
        print(f"  总计: {stats['total_items']} 项")
    
    def save_collection(self, filepath: str):
        """保存收集的数据"""
        data = {
            "target": self.target_person,
            "skill_name": self.skill_name,
            "collected_at": self.collected_at,
            "sources": self.data_sources,
            "statistics": self.get_statistics()
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ 数据已保存到 {filepath}")


if __name__ == "__main__":
    # 示例使用
    collector = InternetDataCollector("Elon Musk", "elon-musk-first-principles")
    collector.print_collection_guide()