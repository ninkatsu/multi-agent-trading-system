"""
Fund Quality Agent - 基金质量分析Agent (基金版"基本面")

职责：评估基金的综合质量——基金经理、穿透持仓、历史业绩、公司实力、规模费率。

与股票版 FundamentalAgent 的核心区别：
- 分析对象从"一家公司"变为"一只基金"（实际是一篮子股票 + 一个基金经理）
- 核心指标从 PE/PB/ROE 变为 经理年限/穿透加权PE/持仓集中度/超额收益
- 新增穿透分析：计算持仓股的加权PE/ROE，评估篮子质量

面试要点：
- 穿透分析是基金分析的"灵魂"——买基金本质是买它底层持有的资产
- 晨星方法论：经理 > 历史业绩 > 费率 > 规模
- 4423法则：快速初筛的经典框架
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.llm import get_llm
from config.json_utils import parse_json_loose
from config.settings import CONFIG
from tools.fund_data import get_provider


@dataclass
class FundQualityAnalysis:
    fund_code: str
    fund_name: str
    fund_type: str
    # 基金经理维度
    manager_score: float
    manager_name: str
    manager_tenure_years: float
    manager_historical_return: float | None
    # 穿透分析维度
    weighted_pe: float | None
    weighted_roe: float | None
    weighted_pb: float | None
    concentration: float  # 前10大集中度
    top3_weight: float
    # 业绩与规模
    fund_size: float  # 亿元
    annual_return_1y: float | None
    excess_return_vs_benchmark: float | None
    # 综合评分
    score: float
    signal: str
    reasoning: str
    # 定投建议
    dca_suitable: bool = False
    pass_4423_filter: bool = False


class FundQualityAgent:
    """基金质量分析Agent：穿透分析 + 基金经理评估 + 4423初筛 → LLM综合评分"""

    SYSTEM_PROMPT = """你是一位资深的公募基金分析师。你的任务是基于以下基金多维数据，给出质量评分和投资建议。

评分维度与标准 (综合1-10分):

一、基金经理 (30%权重):
- 任职本基金 >5年(+3), 3-5年(+2), 1-3年(+1), <1年(0)
- 历史年化 >15%(+3), 10-15%(+2), 5-10%(+1), <5%(0)
- 超额收益 vs 基准: >5%(+2), 0-5%(+1), 跑输(0)

二、持仓穿透质量 (25%权重):
- 加权ROE: >18%(+3), 12-18%(+2), 8-12%(+1), <8%(0)
- 加权PE合理性: <15(+2), 15-25(+1), >30偏贵(-1)
- 集中度前10: 40-60%适中(+2), 30-40%(+1), >80%太高(-1)
- 重仓股中≥3只公认白马(+1)

三、历史业绩 (20%权重):
- 近1年同类排名前25%(+3), 前50%(+2), 前75%(+1), 后25%(-1)
- 近3年年化跑赢基准 >3%(+2), 0-3%(+1), 跑输(-1)
- 最大回撤 <同类(即更好)(+1)

四、基金公司与规模 (15%权重):
- 规模10-100亿适中(+2), 1-10亿或100-300亿(+1), >500亿偏大(0), <1亿(-3清盘风险)
- 头部基金公司(+1)
- 无重大负面(+1)

五、费率 (10%权重):
- 管理费 <1.2%(+1), 1.2-1.5%(0), >1.5%(-1)
- A类持有>2年免赎回费(+1)

类型调整: 指数基金×0.5(不看经理), 债基×0.7(不同评价体系)

请输出JSON:
{
    "score": <1-10的综合评分>,
    "signal": "<BUY/SELL/HOLD>",
    "reasoning": "<150字以内的分析理由>",
    "manager_comment": "<对基金经理的一句话评价>",
    "dca_suitable": <true/false, 是否适合定投>,
    "strengths": ["优势1", "优势2"],
    "weaknesses": ["劣势1", "劣势2"]
}"""

    def __init__(self):
        self.llm = get_llm(temperature=CONFIG.llm.temperature)
        self.provider = get_provider()

    def analyze(self, fund_code: str) -> FundQualityAnalysis:
        info = self.provider.get_fund_info(fund_code)
        look_through = self.provider.look_through_analysis(fund_code)
        manager = self.provider.get_fund_manager(fund_code)
        rank = self.provider.get_fund_rank(fund_code)

        # 4423 快速初筛
        pass_4423 = self._check_4423(fund_code, rank)

        user_prompt = f"""请分析基金 {fund_code} ({info.get('fund_name', '')})：

【基金类型】{info.get('fund_type', '偏股混合型')}
【基金规模】{info.get('fund_size', 'N/A')} 亿元
【成立日期】{info.get('establish_date', 'N/A')}

【基金经理】
- 姓名: {manager.get('name', '未知')}
- 任职年限: {manager.get('tenure_years', 'N/A')}年
- 历史年化: {manager.get('historical_return', 'N/A')}
- 管理总规模: {manager.get('assets_under_mgmt', 'N/A')}亿

【穿透分析（底层持仓加权）】
- 加权PE: {look_through.get('weighted_pe', 'N/A')}
- 加权ROE: {look_through.get('weighted_roe', 'N/A')}
- 加权PB: {look_through.get('weighted_pb', 'N/A')}
- 前10大集中度: {look_through.get('concentration', 'N/A')}
- 前3大占比: {look_through.get('top3_weight', 'N/A')}
- 行业覆盖数: {look_through.get('industry_count', 'N/A')}
- 持仓股票数: {look_through.get('stock_count', 'N/A')}

【费率】
- 管理费: {info.get('management_fee', 'N/A')}
- 托管费: {info.get('custody_fee', 'N/A')}
- 申购费(最高): {info.get('subscribe_fee', 'N/A')}
- 赎回费区间: {info.get('redemption_fee_min', 'N/A')}~{info.get('redemption_fee_max', 'N/A')}

【同类排名】
- 近1年分位: {rank.get('percentile_1y', 'N/A')}

【4423初筛】{'通过' if pass_4423 else '未通过'}

{'(注意: akshare未安装/数据受限，部分数据为占位值，请基于已有数据做保守判断)' if info.get('_warning') else ''}"""

        response = self.llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        result = parse_json_loose(response.content) or {
            "score": 5.0, "signal": "HOLD", "reasoning": "LLM输出解析失败，默认HOLD",
            "manager_comment": "", "dca_suitable": False,
            "strengths": [], "weaknesses": [],
        }

        return FundQualityAnalysis(
            fund_code=fund_code,
            fund_name=str(info.get("fund_name", "")),
            fund_type=str(info.get("fund_type", "")),
            manager_score=float(result.get("manager_score", 5.0)),
            manager_name=str(manager.get("name", "")),
            manager_tenure_years=float(manager.get("tenure_years", 0)),
            manager_historical_return=manager.get("historical_return"),
            weighted_pe=look_through.get("weighted_pe"),
            weighted_roe=look_through.get("weighted_roe"),
            weighted_pb=look_through.get("weighted_pb"),
            concentration=float(look_through.get("concentration", 0)),
            top3_weight=float(look_through.get("top3_weight", 0)),
            fund_size=float(info.get("fund_size", 0)),
            annual_return_1y=None,
            excess_return_vs_benchmark=None,
            score=float(result.get("score", 5.0)),
            signal=str(result.get("signal", "HOLD")),
            reasoning=str(result.get("reasoning", "")),
            dca_suitable=bool(result.get("dca_suitable", False)),
            pass_4423_filter=pass_4423,
        )

    def _check_4423(self, fund_code: str, rank: dict) -> bool:
        """
        4423法则初筛：
        近1年同类前1/4 → 近2年同类前1/4 → 近3年同类前1/2 → 近5年同类前1/3

        简化版: 检查近1年百分位是否在前25%
        """
        pct = rank.get("percentile_1y", 0.5)
        return pct <= 0.25

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph节点入口"""
        fund_code = state["ticker"]  # 复用 ticker 字段存基金代码
        analysis = self.analyze(fund_code)
        return {
            "analyses": [{
                "agent": "fund_quality",
                "ticker": fund_code,
                "score": analysis.score,
                "signal": analysis.signal,
                "reasoning": analysis.reasoning,
                "data": {
                    "fund_name": analysis.fund_name,
                    "fund_type": analysis.fund_type,
                    "manager_name": analysis.manager_name,
                    "manager_tenure_years": analysis.manager_tenure_years,
                    "weighted_pe": analysis.weighted_pe,
                    "weighted_roe": analysis.weighted_roe,
                    "concentration": analysis.concentration,
                    "dca_suitable": analysis.dca_suitable,
                    "pass_4423": analysis.pass_4423_filter,
                    "fund_size": analysis.fund_size,
                },
            }]
        }
