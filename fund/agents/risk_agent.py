"""
Fund Risk Agent - 基金风控守门Agent

职责：按基金类型动态设定风控阈值、费率成本检查、清盘风险警告、T+1流动性约束。

与股票版的核心区别：
- 阈值按基金类型动态设定（偏股/指数/债基/QDII各不相同）
- 新增：清盘风险(<5000万)、费率陷阱(7天内赎回1.5%)、经理变更风险
- 新增：T+1/T+2 流动性约束（不像股票可以紧急止损）
- 止损对基金意义弱（申赎费惩罚+净值平滑），更多用止盈+仓位管理
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from fund.config.llm import get_llm
from fund.config.json_utils import parse_json_loose
from fund.config.settings import CONFIG
from fund.tools.fund_data import get_provider


# 按基金类型的风控阈值
FUND_RISK_RULES = {
    "偏股混合": {"max_position": 0.25, "max_drawdown": 0.12, "stop_loss": 0.10, "min_holding_days": 90},
    "股票型":   {"max_position": 0.20, "max_drawdown": 0.15, "stop_loss": 0.12, "min_holding_days": 90},
    "混合型":   {"max_position": 0.20, "max_drawdown": 0.12, "stop_loss": 0.10, "min_holding_days": 90},
    "指数型":   {"max_position": 0.30, "max_drawdown": 0.15, "stop_loss": 0.12, "min_holding_days": 30},
    "ETF":     {"max_position": 0.30, "max_drawdown": 0.15, "stop_loss": 0.12, "min_holding_days": 7},
    "偏债混合": {"max_position": 0.40, "max_drawdown": 0.06, "stop_loss": 0.04, "min_holding_days": 60},
    "债券型":   {"max_position": 0.50, "max_drawdown": 0.03, "stop_loss": 0.02, "min_holding_days": 30},
    "纯债":     {"max_position": 0.50, "max_drawdown": 0.03, "stop_loss": 0.02, "min_holding_days": 30},
    "QDII":    {"max_position": 0.15, "max_drawdown": 0.15, "stop_loss": 0.12, "min_holding_days": 90},
    "货币型":   {"max_position": 0.60, "max_drawdown": 0.01, "stop_loss": 0.005, "min_holding_days": 0},
}
DEFAULT_RISK = {"max_position": 0.20, "max_drawdown": 0.12, "stop_loss": 0.10, "min_holding_days": 90}


@dataclass
class FundRiskAssessment:
    approved: bool
    risk_score: float
    max_position_allowed: float
    hard_rule_violations: list[str]
    soft_warnings: list[str]
    adjusted_position_pct: float
    holding_period_advice: str
    fund_type: str
    reasoning: str


class FundRiskAgent:
    """基金风控Agent：类型阈值 + 清盘检查 + 费率陷阱 + 流动性约束"""

    SYSTEM_PROMPT = """你是一位严格的风控官，专精国内公募基金风控。你已经看到了辩论结果和硬规则检查结果。

基金风控与股票风控的关键区别:
1. 基金赎回T+1~T+2(甚至T+7)，不能像股票那样紧急止损 → 仓位控制和持有期规划更重要
2. 基金有申赎费 → 频繁操作被费率惩罚 → 决策必须考虑持有期是否匹配费率结构
3. 主动基金的核心风险是"经理离职"和"风格漂移"，而非股价波动
4. 清盘风险(规模<5000万)是基金独有的黑天鹅

请输出JSON:
{
    "approved": true/false,
    "adjusted_position_pct": 0.0-1.0,
    "soft_warnings": ["警告1", "警告2"],
    "holding_period_advice": "short/mid/long",
    "reasoning": "100字风控判断"
}"""

    def __init__(self):
        self.llm = get_llm(temperature=0.1)
        self.provider = get_provider()

    def _classify_fund_type(self, fund_type: str) -> str:
        """将基金类型映射到风控类别"""
        ft = fund_type.lower()
        if "股票" in ft or "偏股" in ft:
            return "偏股混合"
        if "指数" in ft or "etf" in ft:
            return "指数型"
        if "混合" in ft:
            return "混合型"
        if "债券" in ft or "债" in ft:
            return "债券型"
        if "货币" in ft or "现金" in ft:
            return "货币型"
        if "qdi" in ft:
            return "QDII"
        return "混合型"

    def _check_fund_hard_rules(self, fund_code: str, proposed_position: float) -> list[str]:
        """基金专属硬规则检查"""
        violations = []
        info = self.provider.get_fund_info(fund_code)
        fund_type = self._classify_fund_type(info.get("fund_type", ""))
        rules = FUND_RISK_RULES.get(fund_type, DEFAULT_RISK)

        # 1. 仓位检查（按基金类型）
        if proposed_position > rules["max_position"]:
            violations.append(
                f"建议仓位{proposed_position:.1%}超过{fund_type}上限{rules['max_position']:.1%}"
            )

        # 2. 清盘风险
        fund_size = info.get("fund_size", 0)
        if 0 < fund_size < 0.5:
            violations.append(f"基金规模{fund_size:.2f}亿 < 5000万，存在清盘风险！")
        elif 0 < fund_size < 1.0:
            violations.append(f"基金规模{fund_size:.2f}亿 < 1亿，清盘预警")

        # 3. 暂停申购检查
        if "暂停" in str(info.get("subscribe_status", "")):
            violations.append("该基金暂停申购，无法买入")

        # 4. 费率陷阱: 如果建议持有期 < 7天
        min_hold = rules.get("min_holding_days", 90)
        if min_hold < 7:
            violations.append(f"注意: 持有<7天赎回费高达1.5%，建议至少持有30天")

        return violations

    def assess(self, fund_code: str, debate_result: dict) -> FundRiskAssessment:
        proposed_position = debate_result.get("target_position_pct", 0.0)
        info = self.provider.get_fund_info(fund_code)
        fund_type = self._classify_fund_type(info.get("fund_type", ""))
        rules = FUND_RISK_RULES.get(fund_type, DEFAULT_RISK)

        hard_violations = self._check_fund_hard_rules(fund_code, proposed_position)

        if hard_violations:
            return FundRiskAssessment(
                approved=False,
                risk_score=9.0,
                max_position_allowed=rules["max_position"],
                hard_rule_violations=hard_violations,
                soft_warnings=[],
                adjusted_position_pct=0.0,
                holding_period_advice="N/A",
                fund_type=fund_type,
                reasoning=f"硬规则否决: {'; '.join(hard_violations)}",
            )

        user_prompt = f"""请对以下基金投资决策做风控评估：

【基金】{fund_code} ({info.get('fund_name', '')})
【类型】{info.get('fund_type', '')} → 风控类别: {fund_type}
【规模】{info.get('fund_size', 'N/A')} 亿
【辩论结论】{debate_result.get('final_signal', 'HOLD')}
【建议仓位】{proposed_position:.1%}
【建议持有期】{debate_result.get('suggested_holding_period', 'mid')}
【辩论置信度】{debate_result.get('confidence', 0):.1%}
【管理费】{info.get('management_fee', 'N/A')}
【赎回费区间】{info.get('redemption_fee_min', 'N/A')}~{info.get('redemption_fee_max', 'N/A')}

【风控阈值参考】
单基金仓位上限: {rules['max_position']:.0%}
最大回撤容忍: {rules['max_drawdown']:.0%}
建议最短持有: {rules['min_holding_days']}天"""

        response = self.llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        result = parse_json_loose(response.content) or {
            "approved": False, "adjusted_position_pct": 0.0,
            "soft_warnings": ["LLM输出解析失败，保守否决"],
            "holding_period_advice": "mid",
            "reasoning": "解析失败，安全否决",
        }

        adjusted_pos = min(
            result.get("adjusted_position_pct", 0.0),
            rules["max_position"],
        )

        return FundRiskAssessment(
            approved=result.get("approved", False),
            risk_score=5.0,
            max_position_allowed=rules["max_position"],
            hard_rule_violations=[],
            soft_warnings=result.get("soft_warnings", []),
            adjusted_position_pct=adjusted_pos,
            holding_period_advice=result.get("holding_period_advice", "mid"),
            fund_type=fund_type,
            reasoning=result.get("reasoning", ""),
        )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        fund_code = state["ticker"]
        debate_result = state.get("debate_result", {})

        assessment = self.assess(fund_code, debate_result)
        return {
            "risk_assessment": {
                "approved": assessment.approved,
                "risk_score": assessment.risk_score,
                "max_position_allowed": assessment.max_position_allowed,
                "hard_rule_violations": assessment.hard_rule_violations,
                "soft_warnings": assessment.soft_warnings,
                "adjusted_position_pct": assessment.adjusted_position_pct,
                "holding_period_advice": assessment.holding_period_advice,
                "fund_type": assessment.fund_type,
                "reasoning": assessment.reasoning,
            }
        }
