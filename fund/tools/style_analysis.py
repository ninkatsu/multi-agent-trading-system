"""
基于收益的风格分析（Returns-Based Style Analysis, RBSA）

背景 — 持仓黑箱期问题：
  - 中国公募基金每季度公布持仓（季报只披露前10大）
  - 1月1日看到的"最新持仓"数据是9月30日的（已过93天）
  - 这93天里基金经理可能已换掉一半仓位 → 穿透分析严重过时
  - RBSA 通过日收益回归，反向推算出基金当前的"有效持仓风格"
  - 1992年 William Sharpe 提出，是机构基金研究的标准工具

原理：
  基金日收益 = β1×基准1收益 + β2×基准2收益 + ... + α
  → β高 = 基金实际上暴露在该因子/行业上
  → R²高 = 基准能解释基金收益 → 风格稳定
  → R²低 = 基准解释不了 → 经理大幅换仓/调仓了

RBSA vs 季报持仓对比：
  | 方法 | 数据时效 | 精度 | 费用 |
  |------|---------|------|------|
  | 季报穿透 | T-90天 | 高(但已过时) | 免费 |
  | RBSA风格回归 | T-1天 | 中(有误差) | 免费(需基准指数) |
  | 机构定制(晨星) | T-1天 | 高 | ¥¥¥ |

本模块的策略：
  - 基准指数可用时 → 完整 RBSA（回归+漂移检测+黑箱期警告）
  - 基准指数不可用时 → 输出"基于净值曲线的简要风格推断"（仅判断波动/趋势类别）
  - 永远在报告中标注"数据时效性"—这是对用户坦诚
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from fund.tools.fund_data import get_provider


@dataclass
class StyleAnalysisResult:
    fund_code: str
    fund_name: str

    # 回归系数（如果可用）
    betas: dict[str, float]
    alpha_daily: float
    r_squared: float
    tracking_error: float

    # 风格漂移检测
    style_drift_detected: bool
    style_drift_detail: str

    # 核心：数据新鲜度 & 黑箱期警告
    data_freshness: str
    blackout_warning: str  # 最重要的字段 — 坦诚告诉用户数据的局限性

    # 净值简要分析（即使没有基准指数也总能做）
    nav_volatility: float
    nav_trend: str
    nav_drawdown_1y: float

    # 方法说明
    method: str
    confidence: str  # "高" / "中" / "低"


class StyleAnalyzer:
    """
    基金风格分析器

    双路径：
      A) 基准指数可用 → 完整 RBSA 回归
      B) 基准指数不可用 → 基于净值曲线的简要风格判断
          + 明确标注"无法进行RBSA回归，以下分析精度有限"
    """

    def __init__(self):
        self.provider = get_provider()

    def analyze(self, fund_code: str, window: int = 60) -> StyleAnalysisResult:
        info = self.provider.get_fund_info(fund_code)
        fund_name = str(info.get("fund_name", ""))
        fund_type = str(info.get("fund_type", ""))

        # 获取基金每日收益
        fund_returns, nav_last, nav_vol, trend, dd = self._get_fund_stats(fund_code, window)

        if fund_returns is None or len(fund_returns) < 20:
            return StyleAnalysisResult(
                fund_code=fund_code, fund_name=fund_name,
                betas={}, alpha_daily=0, r_squared=0, tracking_error=0,
                style_drift_detected=False, style_drift_detail="数据不足",
                data_freshness="数据不足", blackout_warning="无法进行任何分析",
                nav_volatility=0, nav_trend="未知", nav_drawdown_1y=0,
                method="无", confidence="低",
            )

        # 尝试获取基准指数
        bm_data = self._fetch_benchmarks(fund_returns)

        if bm_data and len(bm_data) >= 2:
            # 路径A：完整RBSA
            return self._full_rbsa(fund_code, fund_name, fund_type, fund_returns, bm_data,
                                   nav_vol, trend, dd, window)
        else:
            # 路径B：仅基于净值分析 + 明确标注精度有限
            return self._nav_only_analysis(fund_code, fund_name, fund_type,
                                           nav_vol, trend, dd, window)

    # ================================================================
    #  路径 A：完整 RBSA
    # ================================================================

    def _full_rbsa(self, code, name, ftype, f_ret, bm_data, vol, trend, dd, w):
        """完整风格回归"""
        bm_df = pd.DataFrame(bm_data)
        aligned = bm_df.copy()
        aligned["fund"] = f_ret.values[-len(bm_df):]
        aligned = aligned.dropna()

        bm_names = list(bm_data.keys())
        try:
            betas, alpha_daily, r2 = self._ols(aligned["fund"].values,
                                               aligned[bm_names].values)
        except Exception:
            betas, alpha_daily, r2 = np.zeros(len(bm_names)), 0.0, 0.0

        betas_dict = {}
        for i, n in enumerate(bm_names):
            betas_dict[n] = round(float(betas[i]), 4)

        # 追踪误差
        te = np.std(aligned["fund"].values - aligned[bm_names[0]].values) * np.sqrt(252)
        if bm_names[0] in aligned.columns:
            pass  # already computed

        # 漂移检测
        drift, drift_detail = self._detect_drift(f_ret.values, aligned[bm_names].values, bm_names)

        # 数据新鲜度 & 黑箱期声明
        from datetime import datetime
        now = datetime.now()
        y, m = now.year, now.month
        # 最近季报截止日
        if m <= 3:
            latest_report = f"{y-1}-09-30"
            lag_days = (now - datetime(y-1, 9, 30)).days
        elif m <= 6:
            latest_report = f"{y}-03-31"
            lag_days = (now - datetime(y, 3, 31)).days
        elif m <= 9:
            latest_report = f"{y}-06-30"
            lag_days = (now - datetime(y, 6, 30)).days
        else:
            latest_report = f"{y}-09-30"
            lag_days = (now - datetime(y, 9, 30)).days

        blackout = (
            f"⚠️ 最新季报持仓截至{latest_report}，距今{lag_days}天。"
            f"这{lag_days}天内经理的操作完全不可见。"
            f"以上RBSA基于最近{w}个交易日收益回归，"
            f"R²={r2:.1%}（{'高于0.7，回归可靠' if r2 > 0.7 else '低于0.5，系数仅供参考' if r2 < 0.5 else '中等'})。"
        )

        return StyleAnalysisResult(
            fund_code=code, fund_name=name,
            betas=betas_dict, alpha_daily=round(float(alpha_daily), 6),
            r_squared=round(float(r2), 4),
            tracking_error=round(float(te), 4),
            style_drift_detected=drift, style_drift_detail=drift_detail,
            data_freshness=f"RBSA: 基于最近{len(aligned)}个交易日日频收益回归（非季报）",
            blackout_warning=blackout,
            nav_volatility=vol, nav_trend=trend, nav_drawdown_1y=dd,
            method="RBSA多元线性回归",
            confidence="高" if r2 > 0.7 else ("中" if r2 > 0.5 else "低"),
        )

    # ================================================================
    #  路径 B：仅净值分析（坦诚标注精度有限）
    # ================================================================

    def _nav_only_analysis(self, code, name, ftype, vol, trend, dd, w):
        """基准指数不可用时，仅基于净值做简要推断"""
        from datetime import datetime
        now = datetime.now()
        y, m = now.year, now.month
        if m <= 3:
            lag_days = (now - datetime(y-1, 9, 30)).days
        elif m <= 6:
            lag_days = (now - datetime(y, 3, 31)).days
        elif m <= 9:
            lag_days = (now - datetime(y, 6, 30)).days
        else:
            lag_days = (now - datetime(y, 9, 30)).days

        blackout = (
            f"⚠️ 投资决策面临三重信息不对称：\n"
            f"  1. 最新季报持仓已滞后~{lag_days}天（基金经理可能已大幅换仓）\n"
            f"  2. 基准指数数据源不可用（akshare东方财富接口连接失败），无法通过日收益回归推断当前风格暴露\n"
            f"  3. 因此以下分析仅基于净值曲线，无法判断基金当前的真实行业/风格敞口\n"
            f"\n"
            f"  我们可以确信的：\n"
            f"  - 净值趋势: {trend}\n"
            f"  - 年化波动率: {vol:.1%}\n"
            f"  - 近1年最大回撤: {dd:.1%}\n"
            f"\n"
            f"  我们不知道的：\n"
            f"  - 基金经理现在持有哪些股票（只有季报截止日的快照）\n"
            f"  - 当前仓位是85%还是60%（季报后已过{lag_days}天）\n"
            f"  - 是否新增了季报中没有的行业暴露\n"
            f"\n"
            f"  建议：结合天天基金App查看盘中估值变化（虽然也不精确）、"
            f"关注基金经理最新路演观点、对比同类基金近期表现来辅助判断。"
        )

        return StyleAnalysisResult(
            fund_code=code, fund_name=name,
            betas={}, alpha_daily=0, r_squared=0, tracking_error=0,
            style_drift_detected=False,
            style_drift_detail="无法进行回归分析（基准指数数据不可用）",
            data_freshness=f"净值分析: 基于最近{w}个交易日",
            blackout_warning=blackout,
            nav_volatility=vol, nav_trend=trend, nav_drawdown_1y=dd,
            method="仅净值曲线分析（非RBSA回归）",
            confidence="低",
        )

    # ================================================================
    #  工具方法
    # ================================================================

    def _get_fund_stats(self, fund_code: str, window: int):
        """从净值历史计算基金收益、波动率、趋势、回撤"""
        nav_df = self.provider.get_nav_history(fund_code, period="1y")
        if nav_df is None or nav_df.empty:
            return None, 0, 0, "未知", 0

        nav_last = 0.0
        vol = 0.0
        trend = "未知"
        dd = 0.0
        fund_returns = None

        if "daily_return" in nav_df.columns:
            fund_returns = nav_df["daily_return"].dropna().tail(window)
            nav_vals = pd.to_numeric(nav_df["unit_nav"] if "unit_nav" in nav_df.columns else nav_df.iloc[:, 1],
                                     errors="coerce").dropna()
        elif "unit_nav" in nav_df.columns:
            nav_vals = pd.to_numeric(nav_df["unit_nav"], errors="coerce").dropna()
        else:
            return None, 0, 0, "未知", 0

        if len(nav_vals) > 0:
            nav_last = float(nav_vals.iloc[-1])

        if fund_returns is None and len(nav_vals) > 1:
            fund_returns = nav_vals.pct_change().dropna().tail(window)

        if fund_returns is not None and len(fund_returns) > 5:
            vol = float(np.std(fund_returns) * np.sqrt(252))

        if len(nav_vals) >= 60:
            sma_60 = float(np.mean(nav_vals.iloc[-60:]))
            trend = "上升" if nav_last > sma_60 * 1.03 else ("下降" if nav_last < sma_60 * 0.97 else "横盘")

        if len(nav_vals) >= 252:
            peak_1y = float(np.max(nav_vals.iloc[-252:]))
            dd = (nav_last - peak_1y) / peak_1y if peak_1y > 0 else 0

        return fund_returns, nav_last, vol, trend, dd

    def _fetch_benchmarks(self, fund_returns) -> dict[str, np.ndarray] | None:
        """
        尝试获取基准指数的日收益率

        级联策略:
          1) akshare 国内指数（东方财富接口，可能不稳定）
          2) yfinance 全球指数回退（^HSI恒生, ^GSPC标普500, 399006.SZ创业板等）
        """
        # ===== 方案1: akshare 国内指数 =====
        tests_ak = {
            "沪深300": "sh000300",
            "创业板": "399006",
        }
        results = {}
        for name, code in tests_ak.items():
            ret = self._get_index_returns_akshare(code, len(fund_returns))
            if ret is not None and len(ret) >= len(fund_returns) * 0.8:
                results[name] = ret

        if len(results) >= 1:
            return results

        # ===== 方案2: yfinance 国际指数回退 =====
        tests_yf = {
            "上证综指": "000001.SS",
            "沪深300ETF": "510300.SS",
            "中国A股": "ASHR",       # Xtrackers Harvest CSI 300 China A-Share ETF (美股)
            "恒生指数": "^HSI",
        }
        for name, ticker in tests_yf.items():
            ret = self._get_index_returns_yfinance(ticker, len(fund_returns))
            if ret is not None and len(ret) >= len(fund_returns) * 0.8:
                results[name] = ret

        return results if len(results) >= 1 else None

    def _get_index_returns_akshare(self, code: str, length: int) -> np.ndarray | None:
        """akshare 获取指数日收益"""
        try:
            import akshare as ak
            df = ak.stock_zh_index_daily_em(symbol=code)
            if df is not None and not df.empty and "close" in df.columns:
                closes = pd.to_numeric(df["close"], errors="coerce").dropna().values
                if len(closes) >= length:
                    return pd.Series(closes).pct_change().dropna().values[-length:]
        except Exception:
            pass
        return None

    def _get_index_returns_yfinance(self, ticker: str, length: int) -> np.ndarray | None:
        """yfinance 获取指数日收益（国际回退方案）"""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            if df is not None and not df.empty and "Close" in df.columns:
                closes = pd.to_numeric(df["Close"], errors="coerce").dropna().values
                if len(closes) >= length:
                    return pd.Series(closes).pct_change().dropna().values[-length:]
        except Exception:
            pass
        return None

    def _ols(self, y: np.ndarray, X: np.ndarray):
        """
        多元线性回归 OLS: y = Xβ + α

        公式:
          [α, β₁, β₂, ...]ᵀ = (ZᵀZ)⁻¹Zᵀy   其中 Z = [1, X]
          R² = 1 - SSE / SST
        """
        n = len(y)
        Z = np.column_stack([np.ones(n), X])
        try:
            coeffs = np.linalg.lstsq(Z, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            coeffs = np.zeros(Z.shape[1])

        alpha = coeffs[0]
        betas = coeffs[1:]
        y_pred = Z @ coeffs
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return betas, alpha, r2

    def _detect_drift(self, fund_ret, bm_ret, bm_names):
        """检测风格漂移：比较前半窗 vs 后半窗回归系数"""
        n = len(fund_ret)
        half = n // 2
        if n < 40 or bm_ret.shape[1] < 1:
            return False, "数据不足，无法检测"

        try:
            b1, _, r1 = self._ols(fund_ret[:half], bm_ret[:half])
        except Exception:
            return False, "前半窗回归失败"

        try:
            b2, _, r2 = self._ols(fund_ret[half:], bm_ret[half:])
        except Exception:
            return False, "后半窗回归失败"

        indicators = []
        for i in range(min(len(b1), len(b2))):
            if abs(b2[i]) > 0.01:
                chg = abs(b2[i] - b1[i]) / max(abs(b2[i]), 0.01)
                if chg > 0.5:
                    name = bm_names[i] if i < len(bm_names) else f"因子{i}"
                    indicators.append(f"{name}暴露变化{chg:.0%}")

        if indicators:
            return True, "；".join(indicators[:3]) + "。持仓可能与季报已不同"
        if r1 > 0.6 and r2 < r1 * 0.6:
            return True, f"拟合度从{r1:.1%}降至{r2:.1%}，经理可能大幅调仓"

        return False, "未检测到漂移"


def run_style_analysis(fund_code: str) -> StyleAnalysisResult:
    return StyleAnalyzer().analyze(fund_code)
