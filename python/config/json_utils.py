"""
容错 JSON 解析工具

为什么需要：GLM 等 LLM 经常把 JSON 包在 ```json ... ``` 围栏里，有时还会在
前后附带解释性文字。原生 json.loads 遇到围栏直接抛 JSONDecodeError，
导致 Agent 走默认回退、整条决策链失真。这里统一做容错：
1) 去掉 markdown 围栏
2) 兜底用正则截取第一个 {...} 块
3) 全部失败返回 None，由调用方决定回退默认值
"""
from __future__ import annotations

import json
import re
from typing import Any


def parse_json_loose(text: Any) -> dict | list | None:
    if text is None:
        return None
    # ChatAnthropic 偶尔把 content 返回成 list[ContentBlock]，统一拍平成字符串
    if isinstance(text, list):
        text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
    if not isinstance(text, str):
        text = str(text)

    s = text.strip()

    # 1) 去掉 ```json / ``` 围栏
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
        s = s.strip()

    # 2) 直接解析
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 3) 兜底：截取第一个 {...} 或 [...] 块
    m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None
