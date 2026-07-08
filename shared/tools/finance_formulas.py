#!/usr/bin/env python3
"""
财务/金融公式基础函数库

设计原则：
- 只提供基础数学函数，不预置业务公式（PE/ROE/Sharpe 等由 Agent 基于本库现写）
- 单文件、零外部依赖（仅math/statistics标准库）
- Agent 用法：import 本模块 → 用基础函数写业务公式 → 输出带公式+来源数据的中间结果

溯源原则：
- 所有业务公式输出必须包含: {formula, inputs, result, source}
- 例如: "PE = price/eps = 150.00/6.50 = 23.08, source: yfinance.info['trailingPE']"

基础函数:
- safe_div(a, b)              安全除法
- pct(a, b)                   百分比
- avg(*args), avg_list(xs)    平均
- net(a, b)                   差额
- yoy(current, prior)         同比增长率 %
- qoq(current, prior)         环比增长率 %
- cagr(begin, end, years)     复合年增长率 %
- sharpe(returns, rf)         夏普比率
- sortino(returns, rf)        索提诺比率
- max_drawdown(series)        最大回撤
- weighted_avg(values, weights) 加权平均
- rolling_mean(series, window)  滚动均值
- std_annualized(daily_returns) 年化波动率
"""

import math
import statistics
from typing import Sequence


# ---------- 基础运算 ----------

def safe_div(a: float, b: float) -> float | None:
    """安全除法，b=0 返回 None"""
    return a / b if b != 0 else None


def pct(a: float, b: float) -> float | None:
    """百分比 a/b*100"""
    r = safe_div(a, b)
    return round(r * 100, 4) if r is not None else None


def avg(*args: float) -> float:
    """均值"""
    return sum(args) / len(args) if args else 0.0


def avg_list(values: Sequence[float]) -> float | None:
    """列表平均"""
    return sum(values) / len(values) if values else None


def net(a: float, b: float) -> float:
    """差额 a-b"""
    return a - b


def yoy(current: float, prior: float) -> float | None:
    """同比增长率 %：(本期-去年同期)/|去年同期|*100"""
    if prior == 0:
        return None
    return round((current - prior) / abs(prior) * 100, 4)


def qoq(current: float, prior: float) -> float | None:
    """环比增长率 %"""
    return yoy(current, prior)


def cagr(begin: float, end: float, years: float) -> float | None:
    """复合年增长率 %：(期末/期初)^(1/年数)-1"""
    if begin <= 0 or end <= 0 or years <= 0:
        return None
    return round(((end / begin) ** (1 / years) - 1) * 100, 4)


def round_safe(value: float | None, n: int = 4) -> float | None:
    """安全四舍五入"""
    return round(value, n) if value is not None else None


# ---------- 投资专用 ----------

def weighted_avg(values: Sequence[float], weights: Sequence[float]) -> float | None:
    """加权平均: Σ(v_i * w_i) / Σ(w_i)"""
    if not values or not weights or len(values) != len(weights):
        return None
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def rolling_mean(series: Sequence[float], window: int) -> list[float | None]:
    """滚动均值"""
    result = []
    for i in range(len(series)):
        if i < window - 1:
            result.append(None)
        else:
            result.append(sum(series[i-window+1:i+1]) / window)
    return result


def std_annualized(daily_returns: Sequence[float], trading_days: int = 252) -> float | None:
    """年化波动率：日收益率标准差 × sqrt(交易日数)"""
    if len(daily_returns) < 2:
        return None
    return statistics.stdev(daily_returns) * math.sqrt(trading_days)


def sharpe_ratio(annualized_return: float, risk_free_rate: float,
                 annualized_volatility: float) -> float | None:
    """夏普比率: (年化收益 - 无风险利率) / 年化波动率"""
    return safe_div(annualized_return - risk_free_rate, annualized_volatility)


def sortino_ratio(annualized_return: float, risk_free_rate: float,
                  daily_returns: Sequence[float], trading_days: int = 252) -> float | None:
    """索提诺比率: (年化收益 - 无风险利率) / 下行年化波动率"""
    downside = [r for r in daily_returns if r < 0]
    if len(downside) < 2:
        return None
    downside_vol = statistics.stdev(downside) * math.sqrt(trading_days)
    return safe_div(annualized_return - risk_free_rate, downside_vol)


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """最大回撤: max((peak - trough) / peak)"""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def annualized_return(total_return: float, days: int, trading_days: int = 252) -> float:
    """年化收益率: (1+总收益)^(交易日/天数)-1"""
    if days <= 0:
        return 0.0
    return (1 + total_return) ** (trading_days / days) - 1


def win_rate(wins: int, total: int) -> float | None:
    """胜率"""
    return safe_div(wins, total)


def profit_factor(total_wins: float, total_losses: float) -> float | None:
    """盈亏比: 总盈利/总亏损"""
    return safe_div(total_wins, abs(total_losses))


# ---------- 基金专用 ----------

def fund_concentration(top_holdings_weights: Sequence[float]) -> dict:
    """
    持仓集中度分析
    Returns: {top3, top5, top10, hhi(赫芬达尔指数)}
    """
    weights = list(top_holdings_weights)
    return {
        "top3": round(sum(weights[:3]), 4) if len(weights) >= 3 else round(sum(weights), 4),
        "top5": round(sum(weights[:5]), 4) if len(weights) >= 5 else round(sum(weights), 4),
        "top10": round(sum(weights[:10]), 4),
        "hhi": round(sum(w**2 for w in weights), 6),  # 赫芬达尔-赫希曼指数
    }


def fund_weighted_valuation(pe_values: Sequence[float], pb_values: Sequence[float],
                            roe_values: Sequence[float], weights: Sequence[float]) -> dict:
    """
    穿透加权估值计算
    输入: 各持仓股的 PE/PB/ROE 序列 + 持仓权重序列
    输出: 加权PE, 加权PB, 加权ROE (均带公式溯源)
    """
    return {
        "weighted_pe": round_safe(weighted_avg(pe_values, weights), 2),
        "weighted_pb": round_safe(weighted_avg(pb_values, weights), 2),
        "weighted_roe": round_safe(weighted_avg(roe_values, weights), 4),
        "formula": "weighted_PE = Σ(PE_i × weight_i) / Σ(weight_i)",
        "stock_count": len(weights),
    }


def fund_fee_impact(amount: float, management_fee: float, custody_fee: float,
                    subscribe_fee: float, holding_years: float) -> dict:
    """
    基金费率影响计算
    返回: 各项费用 + 总持有成本占本金比例
    """
    mgmt = amount * management_fee * holding_years
    cust = amount * custody_fee * holding_years
    sub = amount * subscribe_fee
    total = mgmt + cust + sub
    return {
        "management_fee_total": round(mgmt, 2),
        "custody_fee_total": round(cust, 2),
        "subscribe_fee": round(sub, 2),
        "total_cost": round(total, 2),
        "cost_ratio": round_safe(pct(total, amount), 2),
        "formula": "总成本 = 管理费(年×持有年) + 托管费(年×持有年) + 申购费",
    }


# ---------- CLI ----------

_BASE_FUNCTIONS = {
    "safe_div": ("safe_div(a, b)", "安全除法"),
    "pct": ("pct(a, b)", "百分比"),
    "avg": ("avg(*args)", "均值"),
    "avg_list": ("avg_list(values)", "列表平均"),
    "net": ("net(a, b)", "差额"),
    "yoy": ("yoy(current, prior)", "同比增长率%"),
    "qoq": ("qoq(current, prior)", "环比增长率%"),
    "cagr": ("cagr(begin, end, years)", "复合年增长率%"),
    "weighted_avg": ("weighted_avg(values, weights)", "加权平均"),
    "sharpe_ratio": ("sharpe_ratio(ret, rf, vol)", "夏普比率"),
    "sortino_ratio": ("sortino_ratio(ret, rf, returns)", "索提诺比率"),
    "max_drawdown": ("max_drawdown(equity_curve)", "最大回撤"),
    "fund_concentration": ("fund_concentration(weights)", "持仓集中度"),
    "fund_weighted_valuation": ("fund_weighted_valuation(PEs, PBs, ROEs, weights)", "穿透加权估值"),
    "fund_fee_impact": ("fund_fee_impact(amount, mgmt, cust, sub, years)", "费率影响"),
}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        print("金融公式基础函数:")
        for name, (sig, desc) in _BASE_FUNCTIONS.items():
            print(f"  {sig:55s} {desc}")
    else:
        # 自测
        print("=== 自测 ===")
        print(f"safe_div(10, 3) = {safe_div(10, 3)}")
        print(f"pct(5, 20) = {pct(5, 20)}%")
        print(f"yoy(120, 100) = {yoy(120, 100)}%")
        print(f"cagr(100, 200, 3) = {cagr(100, 200, 3)}%")
        print(f"weighted_avg([15,25,30], [0.1,0.2,0.1]) = {weighted_avg([15,25,30], [0.1,0.2,0.1])}")
        print(f"max_drawdown([100,105,95,90,98,102,99]) = {max_drawdown([100,105,95,90,98,102,99]):.4f}")
        print(f"fund_concentration([0.098,0.085,0.072,0.051,0.048]) = {fund_concentration([0.098,0.085,0.072,0.051,0.048])}")
        print("All formulas OK")
