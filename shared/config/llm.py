"""
LLM 工厂 - 统一构造 LLM 客户端

支持的 provider（由环境变量 LLM_PROVIDER 控制）：
- deepseek (默认): DeepSeek API - 优先 OpenAI 兼容端点 /v1，回退 Anthropic 兼容端点 /anthropic
- openai          : OpenAI 官方或任意 OpenAI 兼容端点
- anthropic       : Anthropic 兼容端点 (GLM / Claude / DeepSeek Anthropic 等)
"""
from __future__ import annotations

import os

from shared.config.settings import CONFIG


def get_llm(temperature: float | None = None, max_tokens: int | None = None):
    """
    根据 LLM_PROVIDER 返回一个可 invoke 的 LLM 客户端。

    Args:
        temperature: 覆盖默认温度；None 则用 settings 里的值
        max_tokens:  覆盖默认最大 token；None 则用 settings 里的值
    """
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    temp = CONFIG.llm.temperature if temperature is None else temperature
    max_tok = CONFIG.llm.max_tokens if max_tokens is None else max_tokens

    if provider == "deepseek":
        # DeepSeek: 优先用 OpenAI 兼容端点 (/v1)，因为更稳定
        base_url = CONFIG.llm.deepseek_base_url
        api_key = CONFIG.llm.deepseek_api_key or CONFIG.llm.anthropic_api_key
        model = CONFIG.llm.deepseek_model

        if "/anthropic" in base_url:
            # DeepSeek Anthropic 兼容端点
            return _create_anthropic(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temp=temp,
                max_tok=max_tok,
            )
        else:
            # DeepSeek OpenAI 兼容端点 (默认 /v1)
            return _create_openai(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temp=temp,
                max_tok=max_tok,
            )

    elif provider == "anthropic":
        return _create_anthropic(
            api_key=CONFIG.llm.anthropic_api_key,
            base_url=CONFIG.llm.anthropic_base_url,
            model=CONFIG.llm.anthropic_model,
            temp=temp,
            max_tok=max_tok,
        )

    else:
        # openai 兼容路径
        base_url = CONFIG.llm.openai_base_url or None
        return _create_openai(
            api_key=CONFIG.llm.openai_api_key,
            base_url=base_url,
            model=CONFIG.llm.openai_model,
            temp=temp,
            max_tok=max_tok,
        )


def _create_openai(api_key: str, base_url: str | None, model: str,
                   temp: float, max_tok: int):
    """创建 OpenAI 兼容客户端 (langchain-openai)"""
    from langchain_openai import ChatOpenAI

    kwargs = dict(
        model=model,
        temperature=temp,
        api_key=api_key,
    )
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def _create_anthropic(api_key: str, base_url: str, model: str,
                      temp: float, max_tok: int):
    """创建 Anthropic 兼容客户端 (langchain-anthropic)"""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model,
        anthropic_api_url=base_url,
        anthropic_api_key=api_key,
        temperature=temp,
        max_tokens=max_tok,
    )
