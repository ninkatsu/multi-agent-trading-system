"""
定投助手模块 — 不依赖持仓，纯基于净值+市场情绪的定投时点判断

定位：每周运行一次，判断"这周该按计划定投、加码、减半、还是暂停"。

不需要知道基金持有什么——只需要知道：
  - 净值在什么位置（贵还是便宜）？
  - 市场是恐惧还是贪婪？
  - 近期的波动和回撤怎么样？

方法论基础：
  - 微笑曲线：净值低时多买，净值高时少买
  - 恐贪指数：别人恐惧时贪婪，别人贪婪时恐惧
  - 成本摊薄：回撤越大，加码效果越好

定投信号（5档）：
  INCREASE  ✅✅ 加码定投（2x）  — 净值低位+市场恐惧+大幅回撤
  NORMAL    ✅   正常定投（1x）  — 净值正常+情绪中性
  REDUCE    ⚠️   减半定投（0.5x）— 净值偏高+情绪偏热
  PAUSE     ⏸️   暂停定投        — 净值极端高位+市场贪婪
  WAIT      ⌛   观望（等信号）    — 数据不足无法判断
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from fund.tools.fund_data import get_provider
from shared.tools.finance_formulas import safe_div, pct, max_drawdown, avg_list


@dataclass
class DcaSignal:
    """定投信号"""
    fund_code: str
    fund_name: str
    check_time: str

    # 核心输出
    signal: str           # INCREASE / NORMAL / REDUCE / PAUSE / WAIT
    score: float          # 0-10 的综合评分（越高越适合定投）

    # 分项得分（可溯源）
    nav_position_score: float       # 净值位置得分（0-3）
    nav_position_detail: str        # 偏离SMA60的百分比
    drawdown_score: float           # 回撤得分（0-3）
    drawdown_detail: str            # 从高点回撤百分比
    sentiment_score: float          # 情绪得分（0-2）
    sentiment_detail: str           # 恐贪指数描述
    volatility_score: float         # 波动率得分（0-2，低波+分）
    volatility_detail: str

    # 数据
    current_nav: float
    sma_60: float
    sma_20: float
    deviation_pct: float           # 偏离SMA60百分比
    drawdown_1y_pct: float         # 近1年回撤
    annual_volatility: float        # 年化波动率
    fear_greed_index: float        # 恐贪指数

    # 建议
    recommended_weekly_amount: str  # 如 "¥2,000/周"
    reasoning: str


class DcaAssistant:
    """
    定投助手

    使用频率：每周一次（建议周末运行）
    纯逻辑判断，不调LLM（速度快，零成本）

    评分体系（总分10分）：

    净值位置 (0-3分)：
      偏离SMA60<-10%（低位）        → 3分（大买点）
      偏离SMA60 -10% ~ -3%          → 2分
      偏离SMA60 -3% ~ +5%（正常）   → 1分
      偏离SMA60 +5% ~ +15%（偏高）  → 0分
      偏离SMA60 >+15%（高位）       → -1分（不计入，但限制总分）

    回撤情况 (0-3分)：
      年内回撤>15%（大回撤）        → 3分（绝佳买点）
      年内回撤10-15%                → 2分
      年内回撤5-10%                 → 1分
      回撤<5%或创新高               → 0分

    市场情绪 (0-2分)：
      恐贪指数<30（恐惧）           → 2分
      恐贪指数30-50（偏恐惧）       → 1分
      恐贪指数50-70（偏贪婪）       → 0分
      恐贪指数>70（贪婪）           → -1分

    波动率 (0-2分)：
      年化波动<20%（低波稳定）      → 2分
      年化波动20-40%                → 1分
      年化波动>40%（高波动）        → 0分（定投不适合超高波？不，反而更好）

    信号映射：
      总分≥7  → INCREASE（加倍定投）
      总分4-6 → NORMAL（正常定投）
      总分1-3 → REDUCE（减半定投）
      总分≤0  → PAUSE（暂停定投）
      数据不足 → WAIT
    """

    def __init__(self):
        self.provider = get_provider()

    def assess(self, fund_code: str, weekly_amount: float = 2000) -> DcaSignal:
        """
        评估当前是否适合定投

        Args:
            fund_code: 基金代码
            weekly_amount: 每周定投金额（元）

        Returns:
            DcaSignal with signal and reasoning
        """
        info = self.provider.get_fund_info(fund_code)
        fund_name = str(info.get("fund_name", ""))

        nav_df = self.provider.get_nav_history(fund_code, period="1y")
        if nav_df is None or nav_df.empty:
            return self._no_data_signal(fund_code, fund_name)

        # 提取净值序列
        nav_vals = None
        if "unit_nav" in nav_df.columns:
            nav_vals = pd.to_numeric(nav_df["unit_nav"], errors="coerce").dropna()
        if nav_vals is None or len(nav_vals) < 60:
            return self._no_data_signal(fund_code, fund_name)

        current_nav = float(nav_vals.iloc[-1])
        sma_20 = float(np.mean(nav_vals.iloc[-20:]))
        sma_60 = float(np.mean(nav_vals.iloc[-60:]))
        deviation = (current_nav - sma_60) / sma_60  # 偏离度

        # === 1. 净值位置评分 (0-3) ===
        nav_score, nav_detail = self._score_nav_position(deviation)

        # === 2. 回撤评分 (0-3) ===
        if len(nav_vals) >= 252:
            peak_1y = float(np.max(nav_vals.iloc[-252:]))
            drawdown = (current_nav - peak_1y) / peak_1y
        elif len(nav_vals) >= 60:
            peak_window = float(np.max(nav_vals))
            drawdown = (current_nav - peak_window) / peak_window
        else:
            drawdown = 0.0
        dd_score, dd_detail = self._score_drawdown(drawdown)

        # === 3. 波动率 ===
        if len(nav_vals) >= 20:
            daily_ret = nav_vals.pct_change().dropna()
            vol = float(np.std(daily_ret.tail(60)) * np.sqrt(252))
        else:
            vol = 0.0
        vol_score, vol_detail = self._score_volatility(vol)

        # === 4. 市场情绪 ===
        fear = self.provider.get_market_fear_greed()
        fg_index = float(fear.get("index", 50))
        sent_score, sent_detail = self._score_sentiment(fg_index, fear.get("label", ""))

        # === 综合评分 ===
        total_score = nav_score + dd_score + sent_score + vol_score

        # 定投信号
        if total_score >= 7:
            signal = "INCREASE"
            multiplier = 2.0
        elif total_score >= 5:
            signal = "NORMAL"
            multiplier = 1.0
        elif total_score >= 3:
            signal = "REDUCE"
            multiplier = 0.5
        elif total_score >= 0:
            signal = "PAUSE"
            multiplier = 0.0
        else:
            signal = "PAUSE"
            multiplier = 0.0

        # 额外规则：净值极端高位即使总分低也要暂停
        if deviation > 0.20:
            signal = "PAUSE"
            multiplier = 0.0

        # 生成金额建议
        amount = weekly_amount * multiplier
        if signal == "INCREASE":
            amount_str = f"¥{amount:,.0f}/周（正常¥{weekly_amount:,.0f}的2倍）"
        elif signal == "NORMAL":
            amount_str = f"¥{weekly_amount:,.0f}/周（正常定投）"
        elif signal == "REDUCE":
            amount_str = f"¥{amount:,.0f}/周（减半）"
        else:
            amount_str = f"¥0/周（暂停，资金保留等待更好时机）"

        # 生成推理
        reasons = []
        if signal == "INCREASE":
            reasons.append("净值处于相对低位且市场恐惧中——这是定投的最佳时机（微笑曲线左侧）")
        elif signal == "PAUSE":
            if deviation > 0.20:
                reasons.append(f"净值偏离均线{deviation:.0%}，严重高估，暂停定投等回调")
            else:
                reasons.append("综合评分低，当前不是好的定投时机")

        return DcaSignal(
            fund_code=fund_code, fund_name=fund_name,
            check_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
            signal=signal, score=round(total_score, 1),
            nav_position_score=nav_score, nav_position_detail=nav_detail,
            drawdown_score=dd_score, drawdown_detail=dd_detail,
            sentiment_score=sent_score, sentiment_detail=sent_detail,
            volatility_score=vol_score, volatility_detail=vol_detail,
            current_nav=current_nav, sma_20=round(sma_20, 4), sma_60=round(sma_60, 4),
            deviation_pct=round(deviation, 4),
            drawdown_1y_pct=round(drawdown, 4),
            annual_volatility=round(vol, 4),
            fear_greed_index=fg_index,
            recommended_weekly_amount=amount_str,
            reasoning="；".join(reasons),
        )

    # ================================================================
    #  评分函数（每个都有明确公式 → 可溯源）
    # ================================================================

    def _score_nav_position(self, deviation: float) -> tuple[float, str]:
        """
        净值位置评分

        公式: score = 3 - clamp((deviation + 0.10) / 0.08, 0, 3)
             deviation = (NAV - SMA60) / SMA60

        deviation < -10%  → 3分（低位买点）
        deviation -10~-3%  → 2分
        deviation -3~+5%   → 1分（正常）
        deviation  +5~15%  → 0分（偏贵）
        deviation > +15%   → 0分（太贵，定投暂停由外层处理）
        """
        if deviation < -0.15:
            return 3.0, f"净值偏离SMA60={deviation:.1%}，深度低位"
        elif deviation < -0.10:
            return 3.0, f"净值偏离SMA60={deviation:.1%}，显著低位"
        elif deviation < -0.03:
            return 2.0, f"净值偏离SMA60={deviation:.1%}，略低于均线"
        elif deviation <= 0.05:
            return 1.5, f"净值偏离SMA60={deviation:.1%}，均线附近（正常）"
        elif deviation <= 0.15:
            return 0.5, f"净值偏离SMA60={deviation:.1%}，高于均线（偏贵）"
        else:
            return 0.0, f"净值偏离SMA60={deviation:.1%}，远高于均线（历史高位）"

    def _score_drawdown(self, dd: float) -> tuple[float, str]:
        """
        回撤评分

        公式: score = 3 - clamp(dd / 0.05, 0, 3)
             dd = (当前净值 - 年内最高) / 年内最高
        """
        if dd < -0.20:
            return 3.0, f"年内回撤{dd:.1%}，深度回撤（绝佳买点）"
        elif dd < -0.15:
            return 3.0, f"年内回撤{dd:.1%}，大幅回撤"
        elif dd < -0.10:
            return 2.0, f"年内回撤{dd:.1%}，中等回撤"
        elif dd < -0.05:
            return 1.0, f"年内回撤{dd:.1%}，轻微回撤"
        elif dd < 0:
            return 0.5, f"年内回撤{dd:.1%}，接近高点"
        else:
            return 0.0, "净值创新高，无回撤（历史高位谨慎买入）"

    def _score_sentiment(self, fg: float, label: str) -> tuple[float, str]:
        """
        市场情绪评分

        公式: score = 2 - clamp((fg - 20) / 25, 0, 2)
             恐惧(<30) → 2分
             中性(30-55) → 1分
             贪婪(>55) → 0分
        """
        if fg < 25:
            return 2.0, f"恐贪指数{fg:.0f}（极度恐惧）— 别人恐惧时贪婪"
        elif fg < 40:
            return 1.5, f"恐贪指数{fg:.0f}（恐惧）— 市场偏悲观，是好时机"
        elif fg < 55:
            return 1.0, f"恐贪指数{fg:.0f}（中性）"
        elif fg < 70:
            return 0.5, f"恐贪指数{fg:.0f}（偏贪婪）— 需谨慎"
        else:
            return 0.0, f"恐贪指数{fg:.0f}（极度贪婪）— 别人贪婪时恐惧"

    def _score_volatility(self, vol: float) -> tuple[float, str]:
        """
        波动率评分 — 高波动对定投反而是优势

        公式: score = min(vol / 0.15, 2.0)
              高波动 → 加分（定投能在低位买到更多）
              低波动 → 减分（定投优势不明显）
        """
        if vol > 0.50:
            return 2.0, f"年化波动{vol:.1%}（极高波动，定投效果最佳）"
        elif vol > 0.35:
            return 1.5, f"年化波动{vol:.1%}（高波动，定投效果显著）"
        elif vol > 0.20:
            return 1.0, f"年化波动{vol:.1%}（中等波动）"
        else:
            return 0.5, f"年化波动{vol:.1%}（低波动，定投优势不明显）"

    # ================================================================
    #  工具
    # ================================================================

    def _no_data_signal(self, code: str, name: str) -> DcaSignal:
        return DcaSignal(
            fund_code=code, fund_name=name,
            check_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
            signal="WAIT", score=0,
            nav_position_score=0, nav_position_detail="数据不足",
            drawdown_score=0, drawdown_detail="数据不足",
            sentiment_score=0, sentiment_detail="数据不足",
            volatility_score=0, volatility_detail="数据不足",
            current_nav=0, sma_20=0, sma_60=0,
            deviation_pct=0, drawdown_1y_pct=0,
            annual_volatility=0, fear_greed_index=50,
            recommended_weekly_amount="数据不足，无法建议",
            reasoning="净值数据不足，等待积累足够历史数据后再运行",
        )


def assess_dca(fund_code: str, weekly_amount: float = 2000) -> DcaSignal:
    """快捷定投评估"""
    return DcaAssistant().assess(fund_code, weekly_amount)


def print_dca_signal(s: DcaSignal):
    """格式化打印定投信号"""
    signal_icons = {
        "INCREASE": "✅✅ 加倍定投",
        "NORMAL": "✅ 正常定投",
        "REDUCE": "⚠️ 减半定投",
        "PAUSE": "⏸️ 暂停定投",
        "WAIT": "⌛ 数据不足",
    }
    print(f"\n{'='*60}")
    print(f"  定投评估: {s.fund_name} ({s.fund_code})")
    print(f"  时间: {s.check_time}")
    print(f"{'='*60}")
    print(f"")
    print(f"  信号: {signal_icons.get(s.signal, s.signal)}")
    print(f"  评分: {s.score}/10")
    print(f"  建议: {s.recommended_weekly_amount}")
    print(f"")
    print(f"  ┌─ 净值位置 ({s.nav_position_score}/3): {s.nav_position_detail}")
    print(f"  ├─ 回撤情况 ({s.drawdown_score}/3): {s.drawdown_detail}")
    print(f"  ├─ 市场情绪 ({s.sentiment_score}/2): {s.sentiment_detail}")
    print(f"  └─ 波动率   ({s.volatility_score}/2): {s.volatility_detail}")
    print(f"")
    print(f"  理由: {s.reasoning}")
    print()


if __name__ == "__main__":
    signal = assess_dca("005844")
    print_dca_signal(signal)
