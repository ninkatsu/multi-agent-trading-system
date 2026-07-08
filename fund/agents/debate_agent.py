"""
Fund Debate Agent - 基金牛熊辩论Agent

职责：接收三维分析结果，从看多(Bull)和看空(Bear)角度辩论，综合裁决。

与股票版的核心区别：
- Prompt 语境从"股票"切换为"基金"
- 辩论角度围绕基金经理、持仓穿透、费率、申赎流动性
- 裁决输出新增：A/C类建议、持有期建议、定投建议
- 辩论机制（2轮+Judge）完全复用

面试要点：
- 这是整个系统中最可复用的组件——对抗性辩论框架与标的关系不大
- 基金版新增的定投/持有期/A-C类的裁决输出是基金专属智能
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.llm import get_llm
from config.json_utils import parse_json_loose
from config.settings import CONFIG

MAX_DEBATE_ROUNDS = 2


@dataclass
class FundDebateResult:
    bull_arguments: list[str]
    bear_arguments: list[str]
    final_signal: str
    confidence: float
    reasoning: str
    recommended_action: str
    target_position_pct: float
    suggested_holding_period: str  # short/mid/long
    a_c_recommendation: str  # A类/C类
    watch_signals: list[str]


class FundDebateAgent:
    """基金辩论Agent：Bull vs Bear + Judge裁决"""

    BULL_PROMPT = """你是一位看好这只基金的买方分析师。请基于以下分析数据，尽全力为「买入/持有」找理由。
即使数据不完美，也要挖掘正面因素。但不要编造数据。

{previous_bear_argument}

请从以下角度组织论点:
1. 基金经理：为什么他/她值得信任（任期、历史业绩、风格稳定性）
2. 穿透持仓：底层资产质量如何（加权ROE、PE合理性、白马股占比）
3. 市场时机：当前是否是好的入场点（净值位置、恐贪指数、定投优势）
4. 比较优势：为什么选这只而不是同类基金或指数

请输出JSON: {{"arguments": ["论点1", "论点2", "论点3"], "confidence": 0-1}}"""

    BEAR_PROMPT = """你是一位谨慎的基金分析师。请基于以下分析数据，尽全力为「不买/卖出」找理由。
找出风险点和被忽视的负面因素。但不要编造数据。

{previous_bull_argument}

请从以下角度组织论点:
1. 基金经理风险：离职风险、风格漂移、规模扩张过快影响操作
2. 持仓风险：集中度偏高、行业暴露、估值偏贵、踩雷可能
3. 费率/流动性：综合费率高、赎回费惩罚、暂停申购风险
4. 替代方案：同类有更好的选择、或直接买指数基金更优

请输出JSON: {{"arguments": ["论点1", "论点2", "论点3"], "confidence": 0-1}}"""

    JUDGE_PROMPT = """你是一位中立的基金投资顾问。你刚刚听完了Bull方和Bear方的辩论。

Bull方论点: {bull_args}
Bear方论点: {bear_args}

基于双方论点和原始分析数据，做出最终裁决。考虑:
- 哪一方的论据更有数据支撑（而非情绪化）？
- 该基金适合什么类型的投资者（稳健/进取）？
- 如果BUY，建议一次性买入还是定投？
- 如果HOLD，等待什么信号出现后重新评估？
- 建议持有期：短期(<6月)/中期(6-24月)/长期(>2年)
- A类还是C类更划算？

请输出JSON:
{{
    "final_signal": "BUY/SELL/HOLD",
    "confidence": 0-1,
    "reasoning": "200字综合裁决理由",
    "recommended_action": "如'定投每周¥2000，连续12周'或'建议先观察，等净值回调至SMA60附近再进'",
    "target_position_pct": 0.0-1.0,
    "suggested_holding_period": "short/mid/long",
    "a_c_recommendation": "A/C",
    "watch_signals": ["信号1", "信号2"]
}}"""

    def __init__(self):
        self.llm = get_llm(temperature=0.5)

    def _format_analyses(self, analyses: list[dict]) -> str:
        lines = []
        for a in analyses:
            lines.append(f"【{a['agent']}分析】评分: {a['score']}/10, 信号: {a['signal']}")
            lines.append(f"  理由: {a['reasoning']}")
            if a.get("data"):
                lines.append(f"  数据: {json.dumps(a['data'], ensure_ascii=False)}")
        return "\n".join(lines)

    def _run_bull(self, data_summary: str, bear_rebuttal: str = "") -> dict:
        prev = f"\nBear方刚才的论点: {bear_rebuttal}\n请反驳并强化你的看多立场。" if bear_rebuttal else ""
        prompt = self.BULL_PROMPT.format(previous_bear_argument=prev)
        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"基金分析数据:\n{data_summary}"),
        ])
        return parse_json_loose(response.content) or {"arguments": ["数据整体偏正面"], "confidence": 0.5}

    def _run_bear(self, data_summary: str, bull_rebuttal: str = "") -> dict:
        prev = f"\nBull方刚才的论点: {bull_rebuttal}\n请反驳并强化你的看空立场。" if bull_rebuttal else ""
        prompt = self.BEAR_PROMPT.format(previous_bull_argument=prev)
        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"基金分析数据:\n{data_summary}"),
        ])
        return parse_json_loose(response.content) or {"arguments": ["存在潜在风险"], "confidence": 0.5}

    def _run_judge(self, data_summary: str, bull_args: list[str], bear_args: list[str]) -> dict:
        prompt = self.JUDGE_PROMPT.format(
            bull_args=json.dumps(bull_args, ensure_ascii=False),
            bear_args=json.dumps(bear_args, ensure_ascii=False),
        )
        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"原始分析数据:\n{data_summary}"),
        ])
        return parse_json_loose(response.content) or {
            "final_signal": "HOLD", "confidence": 0.3,
            "reasoning": "辩论结果解析失败，保守持有",
            "recommended_action": "观望",
            "target_position_pct": 0.0,
            "suggested_holding_period": "mid",
            "a_c_recommendation": "A",
            "watch_signals": ["净值企稳", "资金回流"],
        }

    def debate(self, analyses: list[dict]) -> FundDebateResult:
        data_summary = self._format_analyses(analyses)
        all_bull_args = []
        all_bear_args = []

        bull_result = self._run_bull(data_summary)
        all_bull_args.extend(bull_result.get("arguments", []))

        bear_result = self._run_bear(data_summary, str(all_bull_args))
        all_bear_args.extend(bear_result.get("arguments", []))

        for _ in range(MAX_DEBATE_ROUNDS - 1):
            bull_result = self._run_bull(data_summary, str(all_bear_args))
            all_bull_args.extend(bull_result.get("arguments", []))
            bear_result = self._run_bear(data_summary, str(all_bull_args))
            all_bear_args.extend(bear_result.get("arguments", []))

        judge_result = self._run_judge(data_summary, all_bull_args, all_bear_args)

        return FundDebateResult(
            bull_arguments=all_bull_args,
            bear_arguments=all_bear_args,
            final_signal=judge_result.get("final_signal", "HOLD"),
            confidence=judge_result.get("confidence", 0.3),
            reasoning=judge_result.get("reasoning", ""),
            recommended_action=judge_result.get("recommended_action", "观望"),
            target_position_pct=judge_result.get("target_position_pct", 0.0),
            suggested_holding_period=judge_result.get("suggested_holding_period", "mid"),
            a_c_recommendation=judge_result.get("a_c_recommendation", "A"),
            watch_signals=judge_result.get("watch_signals", []),
        )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        analyses = state.get("analyses", [])
        result = self.debate(analyses)
        return {
            "debate_result": {
                "bull_arguments": result.bull_arguments,
                "bear_arguments": result.bear_arguments,
                "final_signal": result.final_signal,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "recommended_action": result.recommended_action,
                "target_position_pct": result.target_position_pct,
                "suggested_holding_period": result.suggested_holding_period,
                "a_c_recommendation": result.a_c_recommendation,
                "watch_signals": result.watch_signals,
            }
        }
