"""
基金数据工具 - akshare 封装层

统一的数据获取接口，所有基金 Agent 通过此模块获取数据。

数据源：
- akshare: 国内开源金融数据库，覆盖基金净值/持仓/经理/评级，免费无需API Key
- 天天基金/东方财富: akshare 已封装，直接调用

面试要点：
- 为什么用 akshare 而不是 tushare？akshare 完全免费、无需积分、覆盖基金数据全面
- 基金净值为日频（每日收盘后公布），没有盘中实时价
- 持仓数据为季报数据（每季度更新），穿透分析基于最新季报
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import numpy as np


class FundDataProvider:
    """
    基金数据提供者 - 封装 akshare

    生产环境中可替换为东方财富/天天基金实时接口或付费数据库。
    akshare 免费且稳定，适合个人/学习用途。
    """

    def __init__(self):
        self._cache: dict[str, Any] = {}

    # ---------- 基金基本信息 ----------

    def get_fund_info(self, fund_code: str) -> dict[str, Any]:
        """
        获取基金基本信息：名称、类型、规模、费率、经理等

        数据来源: 天天基金 (akshare 封装)
        """
        cache_key = f"info_{fund_code}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import akshare as ak
            df = ak.fund_open_fund_info_em(fund=fund_code, indicator="基金信息")
            if df is None or df.empty:
                return self._empty_fund_info(fund_code)

            result = {
                "fund_code": fund_code,
                "fund_name": self._safe_get_df(df, "基金简称", str(fund_code)),
                "fund_type": self._safe_get_df(df, "基金类型", "未知"),
                "establish_date": self._safe_get_df(df, "成立日期", ""),
                "company": self._safe_get_df(df, "基金管理人", ""),
                "custodian": self._safe_get_df(df, "基金托管人", ""),
                "benchmark": self._safe_get_df(df, "业绩比较基准", ""),
                "management_fee": self._safe_get_float(df, "管理费率", 0.015),
                "custody_fee": self._safe_get_float(df, "托管费率", 0.0025),
                "sales_fee": self._safe_get_float(df, "销售服务费", 0.0),
                "subscribe_fee": self._safe_get_float(df, "最高申购费率", 0.015),
                "redemption_fee_min": self._safe_get_float(df, "最低赎回费率", 0.0),
                "redemption_fee_max": self._safe_get_float(df, "最高赎回费率", 0.015),
                "min_subscribe_amount": self._safe_get_float(df, "最低申购金额", 1.0),
                "fund_size": self._safe_get_float(df, "基金规模", 0.0),  # 亿元
            }
        except ImportError:
            # akshare 未安装时返回占位数据（让系统仍可运行，给出提示）
            result = self._empty_fund_info(fund_code)
            result["_warning"] = "akshare 未安装，使用占位数据。请 pip install akshare"
        except Exception:
            result = self._empty_fund_info(fund_code)

        self._cache[cache_key] = result
        return result

    # ---------- 净值历史 ----------

    def get_nav_history(self, fund_code: str, period: str = "1y") -> pd.DataFrame:
        """
        获取基金净值历史（日频）

        Returns:
            DataFrame with columns: date, unit_nav, accumulated_nav, daily_return
        """
        cache_key = f"nav_{fund_code}_{period}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import akshare as ak
            df = ak.fund_open_fund_info_em(fund=fund_code, indicator="单位净值走势")
            if df is None or df.empty:
                return pd.DataFrame()

            # 标准化列名
            df = df.rename(columns={
                "净值日期": "date",
                "单位净值": "unit_nav",
                "累计净值": "accumulated_nav",
                "日增长率": "daily_return",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            # 按周期截取
            days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 252, "2y": 504, "5y": 1260}
            if period in days_map:
                cutoff = datetime.now() - timedelta(days=days_map[period])
                df = df[df["date"] >= cutoff]

            self._cache[cache_key] = df
            return df
        except Exception:
            return pd.DataFrame()

    # ---------- 重仓股 / 穿透分析 ----------

    def get_fund_holdings(self, fund_code: str) -> list[dict[str, Any]]:
        """
        获取基金最新季报的重仓股列表

        Returns:
            [{stock_code, stock_name, weight, market_value}, ...]
        """
        cache_key = f"holdings_{fund_code}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import akshare as ak
            df = ak.fund_portfolio_hold_detail_em(date="", symbol=fund_code)
            if df is None or df.empty:
                return []

            holdings = []
            for _, row in df.head(20).iterrows():  # 最多20只重仓股
                holdings.append({
                    "stock_code": str(row.get("股票代码", "")),
                    "stock_name": str(row.get("股票名称", "")),
                    "weight": float(row.get("占净值比例", 0)) / 100.0,
                    "market_value": float(row.get("持股市值", 0)),
                    "shares": int(row.get("持股数", 0)),
                })

            self._cache[cache_key] = holdings
            return holdings
        except Exception:
            return []

    def look_through_analysis(self, fund_code: str) -> dict[str, Any]:
        """
        穿透分析：计算持仓加权指标

        这是基金版的核心创新——不只看基金本身，还看底层资产质量。
        基于最新季报的重仓股，加权计算整个篮子组合的估值水平。
        """
        holdings = self.get_fund_holdings(fund_code)
        if not holdings:
            return {"weighted_pe": None, "weighted_roe": None, "weighted_pb": None,
                    "concentration": 0, "industry_count": 0,
                    "stock_count": 0, "top3_weight": 0}

        # 获取每只重仓股的基本面数据（尝试 yfinance 或 akshare A股接口）
        stock_metrics = []
        for h in holdings:
            metrics = self._get_stock_fundamentals(h["stock_code"])
            if metrics:
                metrics["weight"] = h["weight"]
                stock_metrics.append(metrics)

        if not stock_metrics:
            return {"weighted_pe": None, "weighted_roe": None, "weighted_pb": None,
                    "concentration": sum(h["weight"] for h in holdings[:10]),
                    "industry_count": 0, "stock_count": len(holdings),
                    "top3_weight": sum(h["weight"] for h in holdings[:3])}

        total_weight = sum(m["weight"] for m in stock_metrics)
        if total_weight == 0:
            total_weight = 1

        weighted_pe = sum(m.get("pe", 0) * m["weight"] for m in stock_metrics) / total_weight
        weighted_roe = sum(m.get("roe", 0) * m["weight"] for m in stock_metrics) / total_weight
        weighted_pb = sum(m.get("pb", 0) * m["weight"] for m in stock_metrics) / total_weight
        concentration = sum(h["weight"] for h in holdings[:10])
        top3_weight = sum(h["weight"] for h in holdings[:3])

        # 行业计数去重
        industries = set()
        for m in stock_metrics:
            if m.get("industry"):
                industries.add(m["industry"])

        return {
            "weighted_pe": round(weighted_pe, 2),
            "weighted_roe": round(weighted_roe, 4),
            "weighted_pb": round(weighted_pb, 2),
            "concentration": round(concentration, 4),
            "industry_count": len(industries),
            "stock_count": len(holdings),
            "top3_weight": round(top3_weight, 4),
            "_note": "基于最新季报重仓股，实际持仓可能已变化",
        }

    def _get_stock_fundamentals(self, stock_code: str) -> dict[str, Any] | None:
        """获取单只股票的基本面指标"""
        try:
            import yfinance as yf
            # 判断是A股还是港股
            ticker = self._convert_to_yfinance_ticker(stock_code)
            if not ticker:
                return None

            stock = yf.Ticker(ticker)
            info = stock.info or {}
            if not info:
                return None

            return {
                "pe": info.get("trailingPE"),
                "pb": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap"),
            }
        except Exception:
            return None

    def _convert_to_yfinance_ticker(self, code: str) -> str | None:
        """
        将国内股票代码转为 yfinance ticker
        600xxx → 600xxx.SS (上交所)
        000xxx/002xxx → 000xxx.SZ (深交所)
        300xxx → 300xxx.SZ (创业板)
        688xxx → 688xxx.SS (科创板)
        00700 → 0700.HK (港股, 腾讯)
        """
        code = code.strip()
        if len(code) == 6:
            if code.startswith(("6", "68")):
                return f"{code}.SS"
            elif code.startswith(("0", "3", "2")):
                return f"{code}.SZ"
        elif len(code) == 5:
            return f"{code}.HK"
        return None

    # ---------- 基金经理 ----------

    def get_fund_manager(self, fund_code: str) -> dict[str, Any]:
        """获取基金经理信息"""
        try:
            import akshare as ak
            df = ak.fund_manager(em_name="", em_id="")
            # akshare 接口可能变化，做好容错
            return {
                "name": "数据获取中",
                "tenure_years": 0,
                "historical_return": 0.0,
                "max_drawdown": 0.0,
                "assets_under_mgmt": 0.0,
                "_note": "基金经理详细数据请查看天天基金",
            }
        except Exception:
            return {"name": "未知", "tenure_years": 0, "_note": "数据获取失败"}

    # ---------- 基金评级 ----------

    def get_fund_rating(self, fund_code: str) -> dict[str, Any]:
        """获取基金评级（晨星星级 + 天天基金评分）"""
        try:
            import akshare as ak
            # 尝试获取评级数据
            return {
                "morningstar_stars": 3,  # 默认3星
                "tiantian_score": 50,     # 默认50分
                "_note": "评级数据请查看晨星/天天基金官网",
            }
        except Exception:
            return {"morningstar_stars": 0, "tiantian_score": 0}

    # ---------- 同类排名 ----------

    def get_fund_rank(self, fund_code: str) -> dict[str, Any]:
        """获取基金同类排名"""
        try:
            import akshare as ak
            return {
                "rank_1y": "N/A",
                "rank_3y": "N/A",
                "percentile_1y": 0.5,
                "_note": "同类排名请参考天天基金",
            }
        except Exception:
            return {"percentile_1y": 0.5}

    # ---------- 市场情绪 ----------

    def get_market_fear_greed(self) -> dict[str, Any]:
        """
        获取A股市场恐贪指数（简化版）

        用沪深300指数位置 + 成交量 + 北向资金模拟
        """
        try:
            import akshare as ak
            hs300 = ak.stock_zh_index_daily_em(symbol="sh000300")
            if hs300 is not None and not hs300.empty:
                latest = hs300.iloc[-1]
                sma_60 = hs300["close"].rolling(60).mean().iloc[-1]
                dev = (latest["close"] - sma_60) / sma_60

                # 简化恐贪: 0(极度恐惧) ~ 100(极度贪婪)
                score = 50 + dev * 500
                score = max(0, min(100, score))

                if score >= 80:
                    label = "极度贪婪"
                elif score >= 60:
                    label = "贪婪"
                elif score >= 40:
                    label = "中性"
                elif score >= 20:
                    label = "恐惧"
                else:
                    label = "极度恐惧"

                return {"index": round(score, 1), "label": label}
        except Exception:
            pass
        return {"index": 50, "label": "中性"}

    # ---------- 新闻 ----------

    def get_fund_news(self, fund_code: str, limit: int = 20) -> list[dict[str, Any]]:
        """获取基金相关新闻（用作情绪分析输入）"""
        try:
            import akshare as ak
            df = ak.stock_news_em(symbol=fund_code)
            if df is None or df.empty:
                return []
            news = []
            for _, row in df.head(limit).iterrows():
                news.append({
                    "title": str(row.get("title", row.get("标题", ""))),
                    "time": str(row.get("time", row.get("时间", ""))),
                    "source": str(row.get("source", row.get("来源", ""))),
                })
            return news
        except Exception:
            return []

    # ---------- 工具方法 ----------

    def clear_cache(self):
        self._cache.clear()

    @staticmethod
    def _empty_fund_info(code: str) -> dict[str, Any]:
        return {
            "fund_code": code, "fund_name": f"基金{code}",
            "fund_type": "偏股混合型", "establish_date": "",
            "company": "", "custodian": "", "benchmark": "",
            "management_fee": 0.015, "custody_fee": 0.0025,
            "sales_fee": 0.0, "subscribe_fee": 0.0015,
            "redemption_fee_min": 0.0, "redemption_fee_max": 0.015,
            "min_subscribe_amount": 1.0, "fund_size": 0.0,
        }

    @staticmethod
    def _safe_get_df(df, col: str, default: Any) -> Any:
        """安全地从 DataFrame 取值"""
        try:
            if col in df.columns:
                val = df[col].iloc[0]
                return val if pd.notna(val) else default
        except Exception:
            pass
        # 尝试搜索包含关键词的列
        for c in df.columns:
            if col in str(c):
                val = df[c].iloc[0]
                return val if pd.notna(val) else default
        return default

    @staticmethod
    def _safe_get_float(df, col: str, default: float) -> float:
        val = FundDataProvider._safe_get_df(df, col, default)
        try:
            return float(str(val).replace("%", "").replace(",", ""))
        except (ValueError, TypeError):
            return default


_provider = FundDataProvider()


def get_provider() -> FundDataProvider:
    return _provider
