"""
Fund Execution Agent - 基金执行Agent

职责：生成申购/赎回单、A/C类选择、定投计划、15:00时间窗判断。

与股票版的核心区别（全量重写）：
- 股票：限价单/市价单 → 基金：申购单/赎回单（按净值成交）
- 股票：滑点控制 → 基金：申赎费率计算
- 股票：实时成交 → 基金：15:00前按当日净值，之后按次日
- 新增：A/C类选择（按持有期自动选）
- 新增：定投计划生成（基金独有）
- 新增：持有<7天赎回费1.5%惩罚提醒
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fund.config.settings import CONFIG
from fund.tools.fund_data import get_provider


@dataclass
class FundExecutionResult:
    order_id: str
    fund_code: str
    fund_name: str
    order_type: str  # subscribe / redeem / dca_plan / no_action
    amount: float
    share_class: str  # A / C
    nav_date: str  # 按哪天净值成交
    subscribe_fee: float
    estimated_shares: float | None
    status: str
    dca_plan: str | None  # 定投计划
    warnings: list[str]
    message: str


class FundExecutionAgent:
    """基金执行Agent：申赎单 + 定投计划 + A/C选择"""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.provider = get_provider()

    def _determine_share_class(self, holding_period: str) -> str:
        """根据建议持有期选择A类还是C类"""
        if holding_period == "long":
            return "A"  # 长期持有选A(申购费一次性，之后没有销售费)
        elif holding_period == "short":
            return "C"  # 短期持有选C(免申购费，只扣销售费)
        # mid: 6-24月，A类和C类成本接近，默认A
        return "A"

    def _calculate_nav_date(self) -> str:
        """判断按哪天净值成交"""
        now = datetime.now()
        if now.hour < 15:
            return now.strftime("%Y-%m-%d") + " (今日15:00前，按今日净值)"
        else:
            return now.strftime("%Y-%m-%d") + " (已过15:00，按下一交易日净值)"

    def _calculate_subscribe_fee(self, amount: float, share_class: str,
                                  info: dict) -> float:
        """计算申购费"""
        if share_class == "C":
            return 0.0  # C类免申购费
        rate = float(info.get("subscribe_fee", 0.015))
        # 通常互联网平台打1折
        discounted_rate = rate * 0.1
        return round(amount * discounted_rate, 2)

    def _calculate_redemption_warning(self, share_class: str, info: dict) -> str | None:
        """赎回费惩罚提醒"""
        max_redemption = float(info.get("redemption_fee_max", 0))
        if max_redemption >= 0.015:
            return "⚠️ 持有<7天赎回费高达1.5%！建议至少持有30天以上"
        return None

    def execute(self, fund_code: str, risk_assessment: dict,
                debate_result: dict, portfolio_value: float = 1_000_000) -> FundExecutionResult:
        if not risk_assessment.get("approved", False):
            return FundExecutionResult(
                order_id="", fund_code=fund_code, fund_name="",
                order_type="no_action", amount=0, share_class="",
                nav_date="", subscribe_fee=0, estimated_shares=None,
                status="RISK_REJECTED", dca_plan=None,
                warnings=[],
                message=f"风控否决: {risk_assessment.get('reasoning', '')}"
            )

        signal = debate_result.get("final_signal", "HOLD")
        if signal == "HOLD":
            action = debate_result.get("recommended_action", "观望等待")
            return FundExecutionResult(
                order_id="", fund_code=fund_code, fund_name="",
                order_type="no_action", amount=0, share_class="",
                nav_date="", subscribe_fee=0, estimated_shares=None,
                status="NO_ACTION", dca_plan=None,
                warnings=[],
                message=f"决策为HOLD，不执行交易。建议: {action}"
            )

        if signal != "BUY":
            return FundExecutionResult(
                order_id="", fund_code=fund_code, fund_name="",
                order_type="redeem", amount=0, share_class="",
                nav_date="", subscribe_fee=0, estimated_shares=None,
                status="SELL_NOT_SUPPORTED", dca_plan=None,
                warnings=[],
                message="基金版暂不支持自动赎回，请手动操作"
            )

        info = self.provider.get_fund_info(fund_code)
        fund_name = str(info.get("fund_name", ""))
        position_pct = risk_assessment.get("adjusted_position_pct", 0.05)
        holding_period = debate_result.get("suggested_holding_period", "mid")
        share_class = self._determine_share_class(holding_period)
        amount = portfolio_value * position_pct

        nav_date = self._calculate_nav_date()
        subscribe_fee = self._calculate_subscribe_fee(amount, share_class, info)
        current_nav_df = self.provider.get_nav_history(fund_code, period="1mo")
        current_nav = float(current_nav_df["unit_nav"].iloc[-1]) if (current_nav_df is not None and not current_nav_df.empty and "unit_nav" in current_nav_df.columns) else 0
        estimated_shares = (amount - subscribe_fee) / current_nav if current_nav > 0 else None

        warnings = []
        redemption_warning = self._calculate_redemption_warning(share_class, info)
        if redemption_warning:
            warnings.append(redemption_warning)

        if float(info.get("fund_size", 0)) < 1.0:
            warnings.append("⚠️ 基金规模<1亿，注意清盘风险")

        # 定投计划
        dca_plan = None
        recommended_action = debate_result.get("recommended_action", "")
        if "定投" in recommended_action or position_pct < 0.05:
            weeks = 12
            weekly_amount = amount / weeks
            dca_plan = f"建议定投: 每周投入 ¥{weekly_amount:,.0f}，连续{weeks}周，共 ¥{amount:,.0f}"

        order_id = f"FUND-{uuid.uuid4().hex[:8]}"

        if self.dry_run:
            return FundExecutionResult(
                order_id=order_id, fund_code=fund_code, fund_name=fund_name,
                order_type="subscribe_dry_run" if not dca_plan else "dca_plan",
                amount=round(amount, 2),
                share_class=share_class,
                nav_date=nav_date,
                subscribe_fee=subscribe_fee,
                estimated_shares=round(estimated_shares, 2) if estimated_shares else None,
                status="PENDING_DRY_RUN",
                dca_plan=dca_plan,
                warnings=warnings,
                message=f"[模拟] 申购 {fund_name}({fund_code}) {share_class}类 ¥{amount:,.0f} | 申购费¥{subscribe_fee:.2f} | {nav_date}"
            )

        # 真实下单（对接天天基金/支付宝等）
        # 目前国内没有公开的基金交易API，实盘需手动或对接券商
        return FundExecutionResult(
            order_id=order_id, fund_code=fund_code, fund_name=fund_name,
            order_type="subscribe_manual",
            amount=round(amount, 2),
            share_class=share_class,
            nav_date=nav_date,
            subscribe_fee=subscribe_fee,
            estimated_shares=round(estimated_shares, 2) if estimated_shares else None,
            status="REQUIRES_MANUAL_ORDER",
            dca_plan=dca_plan,
            warnings=warnings + ["⚠️ 国内基金暂无公开交易API，请手动在天天基金/支付宝下单"],
            message=f"请在天天基金/支付宝手动申购 {fund_code} {share_class}类 ¥{amount:,.0f}（申购费¥{subscribe_fee:.2f}）"
        )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        fund_code = state["ticker"]
        risk_assessment = state.get("risk_assessment", {})
        debate_result = state.get("debate_result", {})
        portfolio_value = state.get("portfolio_value", 1_000_000)

        result = self.execute(
            fund_code=fund_code,
            risk_assessment=risk_assessment,
            debate_result=debate_result,
            portfolio_value=portfolio_value,
        )
        return {
            "execution_result": {
                "order_id": result.order_id,
                "ticker": result.fund_code,
                "fund_name": result.fund_name,
                "order_type": result.order_type,
                "amount": result.amount,
                "share_class": result.share_class,
                "nav_date": result.nav_date,
                "subscribe_fee": result.subscribe_fee,
                "estimated_shares": result.estimated_shares,
                "dca_plan": result.dca_plan,
                "warnings": result.warnings,
                "status": result.status,
                "message": result.message,
            }
        }
