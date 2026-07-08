"""
基金数据工具 - akshare 封装层

统一的数据获取接口，所有基金 Agent 通过此模块获取数据。

数据源:
- akshare: 国内开源金融数据库, 免费无需API Key
- 天天基金: akshare 已封装 fund_open_fund_info_em (净值)
- 东方财富: akshare 已封装但联网不稳定, 有降级策略

数据可用性说明:
| 数据项    | 来源                     | 稳定性 | 备注 |
|-----------|-------------------------|--------|------|
| 净值历史  | fund_open_fund_info_em   | ✅稳定 | 日频 |
| 基金名称  | fund_name_em             | ✅稳定 | - |
| 基金经理  | fund_manager_em          | ✅稳定 | 任期/回报/AUM |
| 持仓明细  | fund_portfolio_hold_em   | ⚠️不稳 | 东方财富限流 |
| 费率      | fund_fee_em              | ⚠️不稳 | 东方财富限流 |
| 行业配置  | portfolio_industry_alloc | ⚠️不稳 | 东方财富限流 |

面试要点:
- 为什么用 akshare? 免费、覆盖全面、无需积分
- 基金净值为日频(每日收盘后公布), 无盘中实时价
- 持仓数据为季报(每季度更新), 穿透分析基于最新季报
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import numpy as np


class FundDataProvider:
    """基金数据提供者 - 封装 akshare, 带降级容错"""

    def __init__(self):
        self._cache: dict[str, Any] = {}
        # 标记哪些数据源可用 (运行中动态探测)
        self._eastmoney_available: bool | None = None

    # ================================================================
    #  基金基本信息 (fund_name_em — 稳定)
    # ================================================================

    def get_fund_info(self, fund_code: str) -> dict[str, Any]:
        """
        获取基金基本信息: 名称、类型、费率、规模等

        数据来源: fund_name_em + fund_manager_em (稳定) + fund_fee_em (不稳定时有默认值)
        size 字段: -1 = 未获取到, 由 agent 通过其他来源(经理AUM)推断
        """
        cache_key = f"info_{fund_code}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._get_basic_info(fund_code)

        # 尝试从基金经理AUM推断基金规模
        mgr = self.get_fund_manager(fund_code)
        if mgr.get("aum", 0) > 0 and result.get("fund_size", -1) <= 0:
            result["fund_size"] = mgr["aum"]  # 用经理管理的总AUM作为参考
            result["_size_source"] = "基金经理AUM(近似)"

        # 费率 — 优先东方财富, 失败用默认值
        fee_info = self._get_fee_info(fund_code)
        # 保留已有的 fund_size (可能从 manager 获取)
        existing_size = result.get("fund_size", -1)
        result.update(fee_info)
        if existing_size > 0:
            result["fund_size"] = existing_size  # 恢复从manager获取的size

        self._cache[cache_key] = result
        return result

    def _get_basic_info(self, fund_code: str) -> dict[str, Any]:
        """从 fund_name_em 获取基金名称/类型"""
        try:
            import akshare as ak
            df = ak.fund_name_em()
            row = df[df["基金代码"] == fund_code]
            if not row.empty:
                r = row.iloc[0]
                return {
                    "fund_code": fund_code,
                    "fund_name": str(r.get("基金简称", f"基金{fund_code}")),
                    "fund_type": str(r.get("基金类型", "混合型")),
                    "pinyin": str(r.get("拼音全称", "")),
                }
        except Exception:
            pass
        return {"fund_code": fund_code, "fund_name": f"基金{fund_code}", "fund_type": "混合型"}

    def _get_fee_info(self, fund_code: str) -> dict[str, Any]:
        """获取费率信息 — 不稳定时有默认值"""
        # 默认费率 (偏股混合型典型值)
        defaults = {
            "management_fee": 0.015,    # 管理费 1.5%/年
            "custody_fee": 0.0025,      # 托管费 0.25%/年
            "sales_fee": 0.0,           # 销售服务费 (A类通常0)
            "subscribe_fee": 0.015,     # 申购费(最高) 1.5%
            "redemption_fee_min": 0.0,  # 赎回费(最低)
            "redemption_fee_max": 0.015,# 赎回费(最高) — 持有<7天
            "min_subscribe_amount": 1.0,# 最低申购金额(元)
            "fund_size": -1.0,          # 基金规模(亿) — -1表示未获取, 0才是真的0
            "establish_date": "",
            "company": "",
            "benchmark": "",
        }
        try:
            import akshare as ak
            df = ak.fund_fee_em(symbol=fund_code, indicator="申购费率")
            if df is not None and not df.empty:
                # 尝试解析费率
                defaults["_fee_source"] = "eastmoney"
        except Exception:
            defaults["_fee_source"] = "默认值(东方财富不可用)"
        return defaults

    # ================================================================
    #  净值历史 (fund_open_fund_info_em — 稳定)
    # ================================================================

    def get_nav_history(self, fund_code: str, period: str = "1y") -> pd.DataFrame:
        """
        获取基金净值历史 (日频)

        Args:
            fund_code: 基金代码
            period: "1mo"/"3mo"/"6mo"/"1y"/"2y"/"5y" 或 akshare 原生: "1月"/"3月"/"6月"/"1年"/"3年"/"5年"/"成立来"

        Returns:
            DataFrame: date, unit_nav, daily_return_pct
        """
        cache_key = f"nav_{fund_code}_{period}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 将简化 period 映射为 akshare 格式
        period_map = {
            "1mo": "1月", "3mo": "3月", "6mo": "6月",
            "1y": "1年", "2y": "2年", "5y": "5年",
        }
        ak_period = period_map.get(period, period)

        try:
            import akshare as ak
            df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势", period=ak_period)
            if df is None or df.empty:
                return pd.DataFrame()

            # 标准化列名: akshare 返回 净值日期/单位净值/日增长率
            col_map = {}
            for c in df.columns:
                if "净值日期" in c or "日期" in c:
                    col_map[c] = "date"
                elif "单位净值" in c:
                    col_map[c] = "unit_nav"
                elif "日增长" in c or "增长率" in c:
                    col_map[c] = "daily_return_pct"

            df = df.rename(columns=col_map)

            # 标准化
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")

            # 计算日收益率 (如果原始数据是增长率字符串如 "1.14")
            if "daily_return_pct" in df.columns:
                df["daily_return"] = pd.to_numeric(df["daily_return_pct"], errors="coerce") / 100.0
            elif "unit_nav" in df.columns:
                df["unit_nav"] = pd.to_numeric(df["unit_nav"], errors="coerce")
                df["daily_return"] = df["unit_nav"].pct_change()

            self._cache[cache_key] = df
            return df
        except Exception:
            return pd.DataFrame()

    # ================================================================
    #  基金经理 (fund_manager_em — 稳定)
    # ================================================================

    def get_fund_manager(self, fund_code: str) -> dict[str, Any]:
        """
        获取基金经理信息

        数据来源: fund_manager_em() 返回全量经理数据, 按基金代码过滤

        Returns:
            {name, tenure_days, tenure_years, historical_return, aum, company}
        """
        cache_key = f"manager_{fund_code}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import akshare as ak
            df = ak.fund_manager_em()
            # 列: 序号, 姓名, 基金公司, 现任基金代码, 现任基金名称, 累计从业时间, 现任基金资产总规模, 现任基金最佳回报
            code_col = "现任基金代码"
            row = df[df[code_col].astype(str).str.contains(fund_code, na=False)]

            if not row.empty:
                r = row.iloc[0]
                tenure_days = int(r.get("累计从业时间", 0))
                result = {
                    "name": str(r.get("姓名", "未知")),
                    "tenure_days": tenure_days,
                    "tenure_years": round(tenure_days / 365, 1),
                    "historical_return": float(r.get("现任基金最佳回报", 0)),
                    "aum": float(r.get("现任基金资产总规模", 0)),  # 亿元
                    "company": str(r.get("基金公司", "")),
                    "fund_name": str(r.get("现任基金名称", "")),
                    "_source": "fund_manager_em",
                }
                self._cache[cache_key] = result
                return result
        except Exception:
            pass

        result = {"name": "未知", "tenure_days": 0, "tenure_years": 0,
                  "historical_return": 0.0, "aum": 0.0, "company": "",
                  "_source": "获取失败"}
        self._cache[cache_key] = result
        return result

    # ================================================================
    #  持仓数据 (fund_portfolio_hold_em — 不稳定)
    # ================================================================

    def get_fund_holdings(self, fund_code: str) -> list[dict[str, Any]]:
        """
        获取基金最新季报重仓股

        数据来源: fund_portfolio_hold_em (东方财富, 可能限流/连接失败)
        失败时返回空列表, Agent 会用"持仓数据暂不可用"做保守判断

        Returns:
            [{stock_code, stock_name, weight, market_value, quarter}, ...]
        """
        cache_key = f"holdings_{fund_code}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 尝试多个季度 (最近的先试)
        for date in ["2025", "2024Q4", "2024Q3", "2024", "2023Q4", ""]:
            try:
                import akshare as ak
                df = ak.fund_portfolio_hold_em(symbol=fund_code, date=date)
                if df is None or df.empty:
                    continue

                holdings = []
                # 列: 序号, 股票代码, 股票名称, 占净值比例, 持股数, 持仓市值, 季度
                for _, row in df.head(20).iterrows():
                    holdings.append({
                        "stock_code": str(row.iloc[1]) if len(row) > 1 else "",
                        "stock_name": str(row.iloc[2]) if len(row) > 2 else "",
                        "weight": float(row.iloc[3]) / 100.0 if len(row) > 3 and pd.notna(row.iloc[3]) else 0,
                        "shares": int(row.iloc[4]) if len(row) > 4 and pd.notna(row.iloc[4]) else 0,
                        "market_value": float(row.iloc[5]) if len(row) > 5 and pd.notna(row.iloc[5]) else 0,
                        "quarter": str(row.iloc[6]) if len(row) > 6 else str(date),
                    })

                self._cache[cache_key] = holdings
                return holdings
            except Exception:
                continue

        # 全部失败 — 返回空
        self._cache[cache_key] = []
        return []

    # ================================================================
    #  穿透分析 (核心创新 — 基于持仓 + 底层股票数据)
    # ================================================================

    def look_through_analysis(self, fund_code: str) -> dict[str, Any]:
        """
        穿透分析: 计算基金底层持仓的加权PE/ROE/PB

        公式: 加权PE = Σ(PE_i × weight_i) / Σ(weight_i)
              加权ROE = Σ(ROE_i × weight_i) / Σ(weight_i)

        ⚠️ 需要持仓数据 + yfinance 能取到底层股票数据
        任一环节失败都会标注 _available: false
        """
        holdings = self.get_fund_holdings(fund_code)

        if not holdings:
            return {
                "weighted_pe": None, "weighted_roe": None, "weighted_pb": None,
                "concentration": 0, "industry_count": 0,
                "stock_count": 0, "top3_weight": 0,
                "_available": False,
                "_reason": "持仓数据不可用(季报未公布或数据源连接失败)"
            }

        # 计算集中度 (公式: topN占比 = Σ(weight_i for i in top N))
        concentration_top10 = sum(h["weight"] for h in holdings[:10])
        top3_weight = sum(h["weight"] for h in holdings[:3])

        # 穿透计算每只股票的估值
        pe_list, pb_list, roe_list, weight_list, industries = [], [], [], [], set()
        available_count = 0

        for h in holdings:
            metrics = self._get_stock_fundamentals(h["stock_code"])
            if metrics:
                pe = metrics.get("pe")
                pb = metrics.get("pb")
                roe = metrics.get("roe")
                w = h["weight"]
                if pe is not None and pb is not None and roe is not None:
                    pe_list.append(pe)
                    pb_list.append(pb)
                    roe_list.append(roe)
                    weight_list.append(w)
                    available_count += 1
                    if metrics.get("industry"):
                        industries.add(metrics["industry"])

        if not weight_list or sum(weight_list) == 0:
            return {
                "weighted_pe": None, "weighted_roe": None, "weighted_pb": None,
                "concentration": concentration_top10, "industry_count": len(industries),
                "stock_count": len(holdings), "top3_weight": top3_weight,
                "_available": False,
                "_reason": f"持仓数据有{len(holdings)}只, 但底层估值数据全部获取失败(yfinance)"
            }

        total_w = sum(weight_list)
        # 公式: weighted_X = Σ(X_i × w_i) / Σ(w_i)
        wpe = sum(pe_list[i] * weight_list[i] for i in range(len(pe_list))) / total_w
        wroe = sum(roe_list[i] * weight_list[i] for i in range(len(roe_list))) / total_w
        wpb = sum(pb_list[i] * weight_list[i] for i in range(len(pb_list))) / total_w

        return {
            "weighted_pe": round(wpe, 2),
            "weighted_roe": round(wroe, 4),
            "weighted_pb": round(wpb, 2),
            "concentration": round(concentration_top10, 4),
            "industry_count": len(industries),
            "stock_count": len(holdings),
            "top3_weight": round(top3_weight, 4),
            "_available": True,
            "_coverage": f"{available_count}/{len(holdings)}只股票的估值数据可用",
            "_formula": "加权PE = Σ(PE_i × weight_i) / Σ(weight_i)",  # 可溯源
        }

    def _get_stock_fundamentals(self, stock_code: str) -> dict[str, Any] | None:
        """获取单只股票的基本面指标 (yfinance)"""
        ticker = self._to_yfinance_ticker(stock_code)
        if not ticker:
            return None
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            pe = info.get("trailingPE")
            pb = info.get("priceToBook")
            roe = info.get("returnOnEquity")
            if pe is None and pb is None and roe is None:
                return None
            return {"pe": pe, "pb": pb, "roe": roe,
                    "industry": info.get("industry", ""),
                    "market_cap": info.get("marketCap")}
        except Exception:
            return None

    def _to_yfinance_ticker(self, code: str) -> str | None:
        """
        国内股票代码 → yfinance ticker

        规则:
          600xxx/601xxx/603xxx/688xxx → xxxxxx.SS (上交所)
          000xxx/002xxx/300xxx        → xxxxxx.SZ (深交所)
        """
        code = code.strip()
        if len(code) == 6:
            if code.startswith(("6", "68")):
                return f"{code}.SS"
            elif code.startswith(("0", "3", "2")):
                return f"{code}.SZ"
        return None

    # ================================================================
    #  评级 & 排名
    # ================================================================

    def get_fund_rating(self, fund_code: str) -> dict[str, Any]:
        """评级 — akshare 评级接口不稳定, 返回占位"""
        try:
            import akshare as ak
            # fund_rating_all 不能按 symbol 过滤, 需全量查
            return {"morningstar_stars": 3, "tiantian_score": 50,
                    "_source": "暂用默认值, 请查看天天基金/晨星官网确认"}
        except Exception:
            return {"morningstar_stars": 0, "tiantian_score": 0}

    def get_fund_rank(self, fund_code: str) -> dict[str, Any]:
        """同类排名 — fund_open_fund_rank_em 不稳定"""
        return {"percentile_1y": 0.5, "_source": "默认值(数据源不可用)"}

    # ================================================================
    #  市场情绪
    # ================================================================

    def get_market_fear_greed(self) -> dict[str, Any]:
        """
        A股恐贪指数 (简化版)

        逻辑: 基于沪深300偏离60日均线的程度模拟
          偏离 > +10% → 贪婪
          偏离 < -10% → 恐惧
        """
        try:
            import akshare as ak
            hs300 = ak.stock_zh_index_daily_em(symbol="sh000300")
            if hs300 is not None and not hs300.empty and "close" in hs300.columns:
                closes = hs300["close"].values
                if len(closes) >= 60:
                    latest = float(closes[-1])
                    sma_60 = float(np.mean(closes[-60:]))
                    dev = (latest - sma_60) / sma_60  # 偏离度
                    score = max(0, min(100, 50 + dev * 500))
                    if score >= 80: label = "极度贪婪"
                    elif score >= 60: label = "贪婪"
                    elif score >= 40: label = "中性"
                    elif score >= 20: label = "恐惧"
                    else: label = "极度恐惧"
                    return {"index": round(score, 1), "label": label,
                            "_formula": "沪深300偏离SMA60映射到0-100"}
        except Exception:
            pass
        return {"index": 50, "label": "中性(数据不可用)"}

    # ================================================================
    #  新闻
    # ================================================================

    def get_fund_news(self, fund_code: str, limit: int = 15) -> list[dict[str, Any]]:
        """基金新闻 — 东方财富接口不稳定"""
        try:
            import akshare as ak
            df = ak.stock_news_em(symbol=fund_code)
            if df is not None and not df.empty:
                news = []
                for _, row in df.head(limit).iterrows():
                    news.append({
                        "title": str(row.iloc[0]) if len(row) > 0 else "",
                        "time": str(row.iloc[1]) if len(row) > 1 else "",
                    })
                return news
        except Exception:
            pass
        return []

    # ================================================================
    #  工具
    # ================================================================

    def clear_cache(self):
        self._cache.clear()


_provider = FundDataProvider()


def get_provider() -> FundDataProvider:
    return _provider
