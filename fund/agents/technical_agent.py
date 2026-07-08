"""
Fund Technical Agent - 基金技术面分析Agent

职责：分析基金净值走势、均线偏离、相对基准强弱、定投择时信号。

与股票版的核心区别：
- 数据从K线变为日频净值（每天只有一个净值，无盘中价）
- MACD/RSI对基金净值意义减弱，更重SMA趋势和相对基准强弱
- 新增定投择时信号（基金版独有）
- ETF基金可沿用更多股票版指标（有盘中交易价）
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from config.llm import get_llm
from config.json_utils import parse_json_loose
from config.settings import CONFIG
from tools.fund_data import get_provider


@dataclass
class FundTechnicalAnalysis:
    fund_code: str
    current_nav: float
    sma_20: float | None
    sma_60: float | None
    nav_deviation_sma60: float | None  # 偏离60日均线百分比
    vs_benchmark_trend: str  # 跑赢/跑输基准
    volatility: float | None
    dca_signal: str  # 定投信号
    dca_score: float  # 定投评分 0-10
    score: float
    signal: str
    reasoning: str


class FundTechnicalAgent:
    """基金技术面分析：净值趋势 + 定投择时"""

    SYSTEM_PROMPT = """你是一位基金技术分析师。你的任务是分析基金净值走势，给出技术面评分和定投建议。

注意：普通基金只有每日净值（日频），没有盘中K线。技术分析聚焦于：
- 净值趋势（均线排列）
- 相对基准的强弱
- 当前是否适合进场/定投

评分标准 (1-10分):
- 净值在SMA20之上且均线上行: +2
- 净值在SMA60之上（中期趋势向上）: +1
- 净值偏离SMA60 <5%（靠近均线，较好买点）: +2
- 净值偏离SMA60 >15%（短期涨幅过大，谨慎）: -1
- 跑赢业绩基准且趋势向上: +2
- 净值连涨>5天（短期过热）: -1
- 波动率收缩（低位企稳信号）: +1
- 净值创新低（破位）: -2

定投建议 (和评分分开判断):
- 净值在SMA60下方(低位区域): 适合定投
- 近期回撤>10% vs 年内高点: 适合加码定投
- 净值在SMA60上方>10%: 减少定投金额

请输出JSON:
{
    "score": <1-10>,
    "signal": "<BUY/SELL/HOLD>",
    "reasoning": "<100字分析>",
    "dca_signal": "<START/INCREASE/KEEP/REDUCE/STOP>",
    "dca_reason": "<定投建议理由>"
}"""

    def __init__(self):
        self.llm = get_llm(temperature=CONFIG.llm.temperature)
        self.provider = get_provider()

    def compute_indicators(self, fund_code: str) -> dict[str, Any]:
        """基于净值历史计算技术指标"""
        nav_df = self.provider.get_nav_history(fund_code, period="1y")
        if nav_df is None or nav_df.empty or "unit_nav" not in nav_df.columns:
            return {}

        nav = nav_df["unit_nav"].values
        dates = nav_df.index if hasattr(nav_df.index, "strftime") else nav_df["date"]

        # 均线
        sma_20 = float(np.mean(nav[-20:])) if len(nav) >= 20 else float(nav[-1])
        sma_60 = float(np.mean(nav[-60:])) if len(nav) >= 60 else float(nav[-1])
        current_nav = float(nav[-1])

        # 偏离 60 日均线
        nav_dev = (current_nav - sma_60) / sma_60 if sma_60 > 0 else 0

        # 均线趋势
        if len(nav) >= 10:
            sma_20_prev = float(np.mean(nav[-25:-5]))
            sma_60_prev = float(np.mean(nav[-65:-5]))
            trend_direction = "UP" if sma_20 > sma_20_prev and sma_60 > sma_60_prev else "DOWN"
        else:
            trend_direction = "UNKNOWN"

        # 波动率 (20日年化)
        if len(nav) >= 20:
            daily_returns = pd.Series(nav).pct_change().dropna().tail(20)
            volatility = float(daily_returns.std() * np.sqrt(252))
        else:
            volatility = 0.0

        # 年内高点回撤
        year_high = float(np.max(nav[-252:])) if len(nav) >= 20 else current_nav
        drawdown_from_high = (current_nav - year_high) / year_high if year_high > 0 else 0

        # 相对基准强弱 (用沪深300作近似)
        vs_benchmark = self._compare_to_benchmark(nav_df)

        # 定投评分
        dca_score = 0.0
        if nav_dev < 0.05:  # 在均线附近或下方
            dca_score += 3
        if drawdown_from_high < -0.10:  # 回撤>10%
            dca_score += 3
        if -0.05 < nav_dev < 0.05:  # 横盘筑底
            dca_score += 1

        dca_signal = "START" if dca_score >= 6 else ("KEEP" if dca_score >= 4 else "REDUCE")

        return {
            "current_nav": current_nav,
            "sma_20": sma_20,
            "sma_60": sma_60,
            "nav_deviation_sma60": nav_dev,
            "trend_direction": trend_direction,
            "volatility": volatility,
            "drawdown_from_high": drawdown_from_high,
            "vs_benchmark": vs_benchmark,
            "dca_score": dca_score,
            "dca_signal": dca_signal,
        }

    def _compare_to_benchmark(self, nav_df: pd.DataFrame) -> str:
        """简化版相对基准比较"""
        try:
            import akshare as ak
            hs300 = ak.stock_zh_index_daily_em(symbol="sh000300")
            if hs300 is not None and not hs300.empty:
                fund_return = (nav_df["unit_nav"].iloc[-1] - nav_df["unit_nav"].iloc[-60]) / nav_df["unit_nav"].iloc[-60]
                bench_return = (hs300["close"].iloc[-1] - hs300["close"].iloc[-60]) / hs300["close"].iloc[-60]
                if fund_return > bench_return + 0.02:
                    return "OUTPERFORM"
                elif fund_return < bench_return - 0.02:
                    return "UNDERPERFORM"
                return "IN_LINE"
        except Exception:
            pass
        return "UNKNOWN"

    def analyze(self, fund_code: str) -> FundTechnicalAnalysis:
        indicators = self.compute_indicators(fund_code)

        if not indicators:
            return FundTechnicalAnalysis(
                fund_code=fund_code, current_nav=0, sma_20=None, sma_60=None,
                nav_deviation_sma60=None, vs_benchmark_trend="UNKNOWN",
                volatility=None, dca_signal="KEEP", dca_score=5,
                score=5, signal="HOLD", reasoning="数据不足，默认HOLD"
            )

        user_prompt = f"""请分析基金 {fund_code} 的技术面（净值走势）：

- 当前净值: {indicators['current_nav']:.4f}
- SMA(20): {indicators['sma_20']:.4f}
- SMA(60): {indicators['sma_60']:.4f}
- 偏离SMA60: {indicators['nav_deviation_sma60']:.2%}
- 均线趋势: {indicators['trend_direction']}
- 20日波动率(年化): {indicators['volatility']:.2%}
- 年内高点回撤: {indicators['drawdown_from_high']:.2%}
- 相对沪深300: {indicators['vs_benchmark']}
- 定投评分: {indicators['dca_score']:.0f}/10"""

        response = self.llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        result = parse_json_loose(response.content) or {
            "score": 5.0, "signal": "HOLD", "reasoning": "LLM输出解析失败",
            "dca_signal": "KEEP", "dca_reason": ""
        }

        return FundTechnicalAnalysis(
            fund_code=fund_code,
            current_nav=indicators["current_nav"],
            sma_20=indicators["sma_20"],
            sma_60=indicators["sma_60"],
            nav_deviation_sma60=indicators["nav_deviation_sma60"],
            vs_benchmark_trend=indicators["vs_benchmark"],
            volatility=indicators["volatility"],
            dca_signal=result.get("dca_signal", "KEEP"),
            dca_score=indicators["dca_score"],
            score=result.get("score", 5.0),
            signal=result.get("signal", "HOLD"),
            reasoning=result.get("reasoning", ""),
        )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        fund_code = state["ticker"]
        analysis = self.analyze(fund_code)
        return {
            "analyses": [{
                "agent": "technical",
                "ticker": fund_code,
                "score": analysis.score,
                "signal": analysis.signal,
                "reasoning": analysis.reasoning,
                "data": {
                    "current_nav": analysis.current_nav,
                    "sma_60": analysis.sma_60,
                    "nav_deviation": analysis.nav_deviation_sma60,
                    "dca_signal": analysis.dca_signal,
                    "dca_score": analysis.dca_score,
                    "vs_benchmark": analysis.vs_benchmark_trend,
                },
            }]
        }
