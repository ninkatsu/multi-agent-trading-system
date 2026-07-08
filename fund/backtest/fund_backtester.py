"""
Fund Backtester - 基金回测引擎

基于基金净值历史的简化回测，验证策略历史表现。

核心差异 vs 股票版回测：
- 数据源：基金净值（日频）替代股票K线
- 基准：沪深300/基金自身基准 替代 SPY
- 成本模型：申赎费 + 管理费摊销 替代 滑点
- 信号规则：基于净值趋势 + 均线（非MACD/RSI）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fund.tools.fund_data import get_provider


@dataclass
class FundTrade:
    date: str
    fund_code: str
    side: str  # subscribe / redeem
    amount: float
    nav: float
    shares: float
    fee: float


@dataclass
class FundBacktestResult:
    initial_capital: float
    final_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    benchmark_return: float  # 沪深300同期
    excess_return: float
    trades: list[FundTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class FundBacktester:
    """简化基金回测引擎

    信号规则（避免每次调用LLM）:
    - BUY: 净值在SMA60下方 + 近5日横盘 + 跌幅>基准
    - SELL: 净值偏离SMA60 >15% + 涨幅>基准10%以上
    """

    def __init__(self, initial_capital: float = 1_000_000,
                 risk_free_rate: float = 0.03,
                 max_position_pct: float = 0.25,
                 subscribe_fee_rate: float = 0.0015,
                 redemption_penalty_days: int = 30):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.max_position_pct = max_position_pct
        self.subscribe_fee_rate = subscribe_fee_rate
        self.redemption_penalty_days = redemption_penalty_days
        self.provider = get_provider()

    def _generate_signal(self, nav_series: pd.Series, dates: pd.DatetimeIndex,
                          idx: int) -> tuple[str, dict]:
        """简化版信号生成"""
        if idx < 60:
            return "HOLD", {}

        current_nav = float(nav_series.iloc[idx])
        sma_20 = float(nav_series.iloc[idx-20:idx+1].mean())
        sma_60 = float(nav_series.iloc[idx-60:idx+1].mean())
        nav_dev = (current_nav - sma_60) / sma_60

        # 近5日波动判断是否横盘
        recent = nav_series.iloc[idx-5:idx+1]
        range_pct = (float(recent.max()) - float(recent.min())) / float(recent.mean())

        # BUY条件
        buy_signals = 0
        if nav_dev < 0.03:  # 在60日线附近或下方
            buy_signals += 1
        if range_pct < 0.02:  # 横盘
            buy_signals += 1
        if nav_dev < -0.05:  # 显著低于均线
            buy_signals += 1

        # SELL条件
        sell_signals = 0
        if nav_dev > 0.15:  # 偏离均线>15%
            sell_signals += 1
        if nav_dev > 0.10:  # 偏离>10%
            sell_signals += 1

        if buy_signals >= 2:
            return "BUY", {"nav_dev": nav_dev, "sma_60": sma_60, "signal_type": "均值回归"}
        if sell_signals >= 2:
            return "SELL", {"nav_dev": nav_dev, "sma_60": sma_60, "signal_type": "高位止盈"}

        return "HOLD", {}

    def run(self, fund_code: str, start_date: str, end_date: str) -> FundBacktestResult:
        nav_df = self.provider.get_nav_history(fund_code, period="5y")
        if nav_df is None or nav_df.empty or "unit_nav" not in nav_df.columns:
            raise ValueError(f"无法获取基金 {fund_code} 的净值数据")

        nav_df = nav_df.set_index("date")
        # 按日期范围筛选
        mask = (nav_df.index >= start_date) & (nav_df.index <= end_date)
        df = nav_df[mask].copy()
        if df.empty:
            raise ValueError(f"回测区间 {start_date}~{end_date} 无数据")

        nav_series = df["unit_nav"]
        dates = df.index

        cash = self.initial_capital
        shares = 0.0
        trades: list[FundTrade] = []
        equity_curve = []
        wins = []
        losses = []

        last_buy_date = None

        for idx in range(len(df)):
            current_nav = float(nav_series.iloc[idx])
            date_str = str(dates[idx])[:10]
            portfolio_value = cash + shares * current_nav
            equity_curve.append(portfolio_value)

            signal, info = self._generate_signal(nav_series, dates, idx)

            if signal == "BUY" and shares == 0:
                max_invest = portfolio_value * self.max_position_pct
                fee = max_invest * self.subscribe_fee_rate
                invest_amount = max_invest - fee
                new_shares = invest_amount / current_nav
                cash -= max_invest
                shares = new_shares
                last_buy_date = idx
                trades.append(FundTrade(date_str, fund_code, "subscribe",
                                        invest_amount, current_nav, new_shares, fee))

            elif signal == "SELL" and shares > 0:
                sell_value = shares * current_nav
                # 持有<30天有赎回费惩罚
                holding_days = idx - (last_buy_date or 0)
                if holding_days < self.redemption_penalty_days:
                    redemption_fee = sell_value * 0.005  # 0.5%
                else:
                    redemption_fee = 0
                cash += sell_value - redemption_fee

                cost_basis = 0
                for t in trades:
                    if t.fund_code == fund_code and t.side == "subscribe":
                        cost_basis += t.amount

                pnl = sell_value - cost_basis
                if pnl > 0:
                    wins.append(pnl)
                else:
                    losses.append(pnl)
                trades.append(FundTrade(date_str, fund_code, "redeem",
                                        sell_value, current_nav, shares, redemption_fee))
                shares = 0

        # 清仓
        final_nav = float(nav_series.iloc[-1])
        final_value = cash + shares * final_nav

        equity_series = pd.Series(equity_curve)
        if len(equity_series) < 2:
            return FundBacktestResult(
                initial_capital=self.initial_capital, final_value=final_value,
                total_return=0, annualized_return=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown=0, win_rate=0, total_trades=0, winning_trades=0,
                losing_trades=0, avg_win=0, avg_loss=0, profit_factor=0,
                benchmark_return=0, excess_return=0,
            )

        daily_returns = equity_series.pct_change().dropna()
        total_return = (final_value - self.initial_capital) / self.initial_capital
        trading_days = len(df)
        annualized_return = (1 + total_return) ** (252 / max(trading_days, 1)) - 1

        annual_vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0.01
        sharpe = (annualized_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0

        downside = daily_returns[daily_returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else 0.01
        sortino = (annualized_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0

        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = abs(drawdown.min())

        total_wins = len(wins)
        total_losses = len(losses)
        total_trades_count = total_wins + total_losses
        win_rate = total_wins / total_trades_count if total_trades_count > 0 else 0

        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0.01
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")

        # 基准: 沪深300
        benchmark_return = self._get_benchmark_return(start_date, end_date)
        excess_return = total_return - benchmark_return

        return FundBacktestResult(
            initial_capital=self.initial_capital,
            final_value=round(final_value, 2),
            total_return=round(total_return, 4),
            annualized_return=round(annualized_return, 4),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            max_drawdown=round(max_drawdown, 4),
            win_rate=round(win_rate, 4),
            total_trades=len(trades),
            winning_trades=total_wins,
            losing_trades=total_losses,
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            benchmark_return=round(benchmark_return, 4),
            excess_return=round(excess_return, 4),
            trades=trades,
            equity_curve=equity_curve,
        )

    def _get_benchmark_return(self, start_date: str, end_date: str) -> float:
        """获取沪深300同期收益"""
        try:
            import akshare as ak
            hs300 = ak.stock_zh_index_daily_em(symbol="sh000300")
            if hs300 is None or hs300.empty:
                return 0.0
            mask = (hs300["date"] >= start_date) & (hs300["date"] <= end_date)
            period = hs300[mask]
            if len(period) > 1:
                return (float(period["close"].iloc[-1]) - float(period["close"].iloc[0])) / float(period["close"].iloc[0])
        except Exception:
            pass
        return 0.0


def print_fund_backtest_report(result: FundBacktestResult):
    """打印回测报告"""
    print("=" * 60)
    print("             基金回测绩效报告")
    print("=" * 60)
    print(f"  初始资金:        ¥{result.initial_capital:,.0f}")
    print(f"  最终市值:        ¥{result.final_value:,.0f}")
    print(f"  总收益率:        {result.total_return:.2%}")
    print(f"  年化收益率:      {result.annualized_return:.2%}")
    print(f"  基准(沪深300):   {result.benchmark_return:.2%}")
    print(f"  超额收益:        {result.excess_return:.2%}")
    print("-" * 60)
    print(f"  夏普比率:        {result.sharpe_ratio}")
    print(f"  索提诺比率:      {result.sortino_ratio}")
    print(f"  最大回撤:        {result.max_drawdown:.2%}")
    print(f"  盈亏比:          {result.profit_factor}")
    print("-" * 60)
    print(f"  总交易次数:      {result.total_trades}")
    print(f"  胜率:            {result.win_rate:.2%}")
    print(f"  盈利交易:        {result.winning_trades}")
    print(f"  亏损交易:        {result.losing_trades}")
    print(f"  平均盈利:        ¥{result.avg_win:,.2f}")
    print(f"  平均亏损:        ¥{result.avg_loss:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    bt = FundBacktester(initial_capital=1_000_000)
    result = bt.run("005827", "2024-01-01", "2025-06-30")
    print_fund_backtest_report(result)
