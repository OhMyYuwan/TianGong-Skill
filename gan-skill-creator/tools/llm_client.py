"""
LLM 客户端封装：统一的 LLM 调用接口
支持多种后端：Claude API、本地模型等
"""
from anthropic import Anthropic
from typing import Dict, List, Optional, Union
import json
import os


class LLMClient:
    """
    统一的 LLM 客户端接口

    支持的后端：
    - anthropic: Claude API（默认）
    - openai: OpenAI API（可选）
    - local: 本地模型（可选）
    """

    def __init__(
        self,
        backend: str = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.backend = backend

        if backend == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=self.api_key)
            self.model = model or "claude-3-5-sonnet-20241022"

        elif backend == "openai":
            # 可选支持
            try:
                import openai
                self.client = openai.Client(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
                self.model = model or "gpt-4"
            except ImportError:
                raise ImportError("OpenAI 后端需要安装 openai: pip install openai")

        else:
            raise ValueError(f"不支持的后端: {backend}")

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        json_mode: bool = False
    ) -> str:
        """
        执行补全请求

        Args:
            prompt: 用户提示
            system: 系统提示（可选）
            max_tokens: 最大输出 token 数
            temperature: 温度参数
            json_mode: 是否要求 JSON 输出

        Returns:
            模型输出的文本
        """
        if self.backend == "anthropic":
            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": messages
            }

            if system:
                kwargs["system"] = system

            response = self.client.messages.create(**kwargs)
            return response.content[0].text

        elif self.backend == "openai":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

    def complete_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2000
    ) -> Dict:
        """
        执行请求并返回 JSON

        Args:
            prompt: 用户提示
            system: 系统提示
            max_tokens: 最大输出 token 数

        Returns:
            解析后的 JSON 对象
        """
        # 在提示中明确要求 JSON
        json_prompt = f"{prompt}\n\n请只返回 JSON，不要其他文字。"

        response = self.complete(
            prompt=json_prompt,
            system=system,
            max_tokens=max_tokens
        )

        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取 JSON 块
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"无法解析 JSON: {response[:200]}...")

    def count_tokens(self, text: str) -> int:
        """
        估算文本的 token 数

        Args:
            text: 要估算的文本

        Returns:
            估算的 token 数
        """
        # 简单估算：平均每 4 个字符 = 1 token
        # 这是一个粗略估计，实际应使用 tiktoken 等
        return len(text) // 4


# 便捷函数
def get_client(backend: str = "anthropic", **kwargs) -> LLMClient:
    """获取 LLM 客户端实例"""
    return LLMClient(backend=backend, **kwargs)


if __name__ == "__main__":
    # 测试
    client = LLMClient()

    # 简单测试
    response = client.complete("你好，请用一句话介绍自己。")
    print(f"响应: {response}")

    # JSON 测试
    json_response = client.complete_json(
        "请返回一个包含 name 和 age 字段的 JSON 对象，代表一个虚拟人物。"
    )
    print(f"JSON 响应: {json_response}")
