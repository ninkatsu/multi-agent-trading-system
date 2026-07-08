"""
LLM 工厂 - 统一构造 LLM 客户端

为什么需要工厂（而不是在每个 Agent 里直接 `ChatOpenAI(...)`）？
- 统一接线：切换 provider / 模型 / 端点只改环境变量，业务代码零改动
- 集中处理 base_url / auth_token 等差异，Agent 只关心业务 prompt
- 墙内默认走智谱 GLM 的 Anthropic 兼容端点（国内直连，无需代理）

支持的 provider（由环境变量 LLM_PROVIDER 控制）：
- anthropic（默认）: 走 GLM / Claude 等任何 Anthropic 兼容端点
- openai          : 走 OpenAI 官方或任意 OpenAI 兼容端点
"""
from __future__ import annotations

import os

from config.settings import CONFIG


def get_llm(temperature: float | None = None, max_tokens: int | None = None):
    """
    根据 LLM_PROVIDER 返回一个可 invoke 的 LLM 客户端。

    Args:
        temperature: 覆盖默认温度；None 则用 settings 里的值
        max_tokens:  覆盖默认最大 token；None 则用 settings 里的值
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    temp = CONFIG.llm.temperature if temperature is None else temperature
    max_tok = CONFIG.llm.max_tokens if max_tokens is None else max_tokens

    if provider == "anthropic":
        # 智谱 GLM 的 Anthropic 兼容端点。
        # 用 anthropic_api_url + anthropic_api_key 接线：
        # - anthropic_api_url 设定端点（对应 ANTHROPIC_BASE_URL）
        # - anthropic_api_key 走 x-api-key 头（实测智谱端点接受该鉴权方式）
        # 注意：不要传 client= 参数，langchain-anthropic 1.x 会把它当成
        # 请求体 kwarg 转发给 messages.create 而报错。
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "glm-4.5"),
            anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
            anthropic_api_key=os.getenv("ANTHROPIC_AUTH_TOKEN"),
            temperature=temp,
            max_tokens=max_tok,
        )

    # openai 兼容路径（OpenAI 官方 / DeepSeek / 智谱 OpenAI 端点等）
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=CONFIG.llm.model,
        temperature=temp,
        api_key=CONFIG.llm.api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
