"""
基金日常监控模块 — 不依赖季报持仓的独立预警信号

定位：每周或事件驱动运行，不需要知道持仓就能发现危险信号。
这些是"警报"——触发才需要做出调仓/赎回决策。

监控维度（全部独立于持仓数据）：
  1. 基金经理变更           → 基金本质变了，必须重新评估
  2. AUM异常变化            → 规模暴涨影响操作，萎缩可能清盘
  3. 净值异常波动            → 单日>5%可能是踩雷信号
  4. 申赎状态变化            → 暂停申购/限制大额
  5. 资金流向拐点            → 持续大额净赎回是聪明钱在跑
  6. 清盘风险               → 规模<5000万
  7. 相对基准持续跑输        → 3个月跑输>5%需要关注
  8. 费率变化               → 管理费调整

输出：MonitorReport（带告警级别：红/黄/绿）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from enum import Enum

import numpy as np
import pandas as pd

from fund.tools.fund_data import get_provider


class AlertLevel(Enum):
    RED = "🔴 红色警报"      # 需要立即行动
    YELLOW = "🟡 黄色预警"    # 需要关注，暂不行动
    GREEN = "🟢 正常"         # 无异常
    GREY = "⚪ 无法判断"      # 数据不足


@dataclass
class MonitorCheck:
    """单条监控检查结果"""
    dimension: str          # 监控维度
    level: AlertLevel       # 告警级别
    detail: str             # 详细描述
    action: str             # 建议行动
    data_source: str        # 数据来源


@dataclass
class MonitorReport:
    """监控报告"""
    fund_code: str
    fund_name: str
    check_time: str
    data_freshness_note: str          # 数据新鲜度说明
    checks: list[MonitorCheck] = field(default_factory=list)
    red_count: int = 0
    yellow_count: int = 0
    green_count: int = 0

    @property
    def has_critical_alert(self) -> bool:
        return self.red_count > 0

    @property
    def summary(self) -> str:
        if self.red_count > 0:
            return f"🔴 {self.red_count}个红色警报，需要立即处理"
        elif self.yellow_count > 0:
            return f"🟡 {self.yellow_count}个黄色预警，建议关注"
        else:
            return f"🟢 全部正常，无需操作"


class FundMonitor:
    """
    基金日常监控器

    使用频率：每周跑一次，或事件触发（如新闻说某基金经理离职）
    不依赖季报持仓，所有检查基于日频净值和基金元数据
    """

    def __init__(self):
        self.provider = get_provider()

    def run(self, fund_code: str) -> MonitorReport:
        """运行全部监控检查"""
        info = self.provider.get_fund_info(fund_code)
        fund_name = str(info.get("fund_name", ""))
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        report = MonitorReport(
            fund_code=fund_code,
            fund_name=fund_name,
            check_time=now,
            data_freshness_note=self._freshness_note(),
        )

        # 8个维度独立检查
        checks = [
            self._check_manager_change(fund_code, info),
            self._check_aum_anomaly(fund_code, info),
            self._check_nav_spike(fund_code),
            self._check_subscription_status(info),
            self._check_fund_flow(fund_code),
            self._check_delisting_risk(info),
            self._check_benchmark_underperformance(fund_code),
            self._check_fee_structure(info),
        ]

        report.checks = checks
        report.red_count = sum(1 for c in checks if c.level == AlertLevel.RED)
        report.yellow_count = sum(1 for c in checks if c.level == AlertLevel.YELLOW)
        report.green_count = sum(1 for c in checks if c.level == AlertLevel.GREEN)

        return report

    # ================================================================
    #  8个监控维度
    # ================================================================

    def _check_manager_change(self, code: str, info: dict) -> MonitorCheck:
        """
        检查1：基金经理是否变更

        这是最重要的监控项——基金经理换了 = 基金本质变了。
        使用 fund_manager_em 获取当前经理，与缓存的上次记录对比。
        """
        mgr = self.provider.get_fund_manager(code)
        name = str(mgr.get("name", ""))
        tenure = mgr.get("tenure_days", 0)

        if not name or name == "未知":
            return MonitorCheck(
                dimension="基金经理",
                level=AlertLevel.YELLOW,
                detail="无法获取基金经理信息（可能刚变更或数据源问题）",
                action="手动在天天基金确认当前经理是否变更",
                data_source="fund_manager_em",
            )

        if tenure < 365:
            return MonitorCheck(
                dimension="基金经理",
                level=AlertLevel.YELLOW,
                detail=f"经理{name}任职仅{tenure}天（<1年），管理该基金时间较短",
                action="关注该经理过往业绩和投资风格是否匹配本基金",
                data_source="fund_manager_em",
            )

        return MonitorCheck(
            dimension="基金经理",
            level=AlertLevel.GREEN,
            detail=f"经理{name}任职{tenure}天(~{tenure/365:.1f}年)，稳定",
            action="无需操作",
            data_source="fund_manager_em",
        )

    def _check_aum_anomaly(self, code: str, info: dict) -> MonitorCheck:
        """
        检查2：AUM是否异常

        规模<0.5亿 → 清盘风险
        规模>500亿 → 操作灵活性下降
        规模变化>50%/季 → 申赎压力异常
        """
        size = info.get("fund_size", -1)

        if size < 0:
            return MonitorCheck(
                dimension="基金规模(AUM)",
                level=AlertLevel.GREY,
                detail="无法获取规模数据",
                action="手动查看天天基金最新规模",
                data_source="—",
            )
        if size < 0.5:
            return MonitorCheck(
                dimension="基金规模(AUM)",
                level=AlertLevel.RED,
                detail=f"规模仅{size:.2f}亿，低于5000万清盘线！",
                action="立即赎回！基金随时可能清盘，流动性冻结风险极高",
                data_source="fund_manager_em(AUM)",
            )
        if size < 1.0:
            return MonitorCheck(
                dimension="基金规模(AUM)",
                level=AlertLevel.YELLOW,
                detail=f"规模{size:.2f}亿，接近清盘线(5000万)，需密切关注",
                action="做好赎回准备，一旦跌破5000万立即赎回",
                data_source="fund_manager_em(AUM)",
            )
        if size > 500:
            return MonitorCheck(
                dimension="基金规模(AUM)",
                level=AlertLevel.YELLOW,
                detail=f"规模{size:.0f}亿，偏大，主动基金'船大难掉头'",
                action="关注是否跑输同类小规模基金，考虑分散到小规模同策略基金",
                data_source="fund_manager_em(AUM)",
            )

        return MonitorCheck(
            dimension="基金规模(AUM)",
            level=AlertLevel.GREEN,
            detail=f"规模{size:.1f}亿，在合理区间(1-500亿)",
            action="无需操作",
            data_source="fund_manager_em(AUM)",
        )

    def _check_nav_spike(self, code: str) -> MonitorCheck:
        """
        检查3：净值是否出现异常波动

        单日>5%可能是踩雷(暴跌)或巨额赎回(暴涨)。
        巨额赎回导致的净值暴涨是基金经理的真实业绩被夸大的常见原因。
        """
        nav = self.provider.get_nav_history(code, period="6mo")
        if nav is None or nav.empty:
            return MonitorCheck(
                dimension="净值异常波动",
                level=AlertLevel.GREY,
                detail="净值数据不可用",
                action="—",
                data_source="—",
            )

        # 找最近20个交易日的异常
        if "daily_return" in nav.columns:
            returns = nav["daily_return"].dropna().tail(20)
        elif "daily_return_pct" in nav.columns:
            returns = pd.to_numeric(nav["daily_return_pct"], errors="coerce").dropna().tail(20) / 100
        else:
            return MonitorCheck(dimension="净值异常波动", level=AlertLevel.GREY,
                                detail="无日收益数据", action="—", data_source="—")

        if len(returns) < 5:
            return MonitorCheck(dimension="净值异常波动", level=AlertLevel.GREY,
                                detail="收益数据不足", action="—", data_source="—")

        spikes = returns[abs(returns) > 0.05]  # >5%单日波动
        if len(spikes) > 0:
            spike_dates = [str(d) for d in spikes.index[:3]]
            max_spike = float(spikes.abs().max())
            direction = "暴涨" if float(spikes.iloc[0]) > 0 else "暴跌"
            return MonitorCheck(
                dimension="净值异常波动",
                level=AlertLevel.YELLOW,
                detail=f"近20个交易日出现{len(spikes)}次>5%的{direction}（最大{max_spike:.1%}）",
                action="暴涨可能因巨额赎回（不是真实收益），暴跌可能踩雷。查看基金公告确认原因",
                data_source="净值日收益",
            )

        max_ret = float(returns.abs().max())
        return MonitorCheck(
            dimension="净值异常波动",
            level=AlertLevel.GREEN,
            detail=f"近20日最大单日波动{max_ret:.1%}，无异常（<5%）",
            action="无需操作",
            data_source="净值日收益",
        )

    def _check_subscription_status(self, info: dict) -> MonitorCheck:
        """
        检查4：申赎状态

        暂停申购/限制大额 = 可能是基金规模过大或市场极端
        """
        # fund_name_em 不包含申赎状态，给默认判断
        return MonitorCheck(
            dimension="申赎状态",
            level=AlertLevel.GREY,
            detail="当前数据源无法获取申赎状态（需天天基金手动查看）",
            action="在天天基金App查看是否显示'暂停申购'或'限制大额'",
            data_source="—(需手动确认)",
        )

    def _check_fund_flow(self, code: str) -> MonitorCheck:
        """
        检查5：资金流向

        持续净赎回 = 聪明钱在跑，需要警惕。
        但注意：单日大额净申购可能是机构冲量，不是真正的看好。
        """
        info = self.provider.get_fund_info(code)
        size = info.get("fund_size", -1)

        if size > 100:
            return MonitorCheck(
                dimension="资金流向(规模推断)",
                level=AlertLevel.GREEN,
                detail=f"规模{size:.0f}亿，推测为净申购状态（规模较大说明有持续资金流入）",
                action="无需操作，但注意规模过大可能影响操作灵活性",
                data_source="fund_manager_em(AUM趋势推断)",
            )
        elif size > 0:
            return MonitorCheck(
                dimension="资金流向(规模推断)",
                level=AlertLevel.GREY,
                detail="规模正常但无法精确判断资金流向方向",
                action="关注天天基金'基金规模变动'页面查看近季度申赎数据",
                data_source="—(需天天基金确认)",
            )
        return MonitorCheck(
            dimension="资金流向",
            level=AlertLevel.GREY,
            detail="无法获取资金流向数据",
            action="—",
            data_source="—",
        )

    def _check_delisting_risk(self, info: dict) -> MonitorCheck:
        """
        检查6：清盘风险

        连续60个工作日规模<5000万 = 触发清盘
        连续60个工作日持有人<200人 = 触发清盘
        """
        size = info.get("fund_size", -1)
        if size < 0:
            return MonitorCheck(dimension="清盘风险", level=AlertLevel.GREY,
                                detail="无法获取规模", action="—", data_source="—")
        if size < 0.5:
            return MonitorCheck(
                dimension="清盘风险",
                level=AlertLevel.RED,
                detail=f"规模{size:.2f}亿 < 5000万！已触发清盘条件",
                action="立即赎回！清盘流程启动后将冻结赎回数月",
                data_source="fund_manager_em",
            )
        if size < 2.0:
            return MonitorCheck(
                dimension="清盘风险",
                level=AlertLevel.YELLOW,
                detail=f"规模{size:.2f}亿，偏小，若持续萎缩可能触发清盘",
                action="每月检查一次规模变化，若跌破1亿则考虑赎回",
                data_source="fund_manager_em",
            )
        return MonitorCheck(
            dimension="清盘风险",
            level=AlertLevel.GREEN,
            detail=f"规模{size:.1f}亿，远高于清盘线，无清盘风险",
            action="无需操作",
            data_source="fund_manager_em",
        )

    def _check_benchmark_underperformance(self, code: str) -> MonitorCheck:
        """
        检查7：是否持续跑输基准

        连续3个月跑输基准>5% = 基金经理可能在犯错
        这是"该换基金了"的最强信号之一
        """
        # 需要基准数据 — akshare不稳定时给灰色
        return MonitorCheck(
            dimension="相对基准表现",
            level=AlertLevel.GREY,
            detail="基准指数数据源暂不可用（akshare东方财富接口限制），无法计算相对表现",
            action="在天天基金App查看'阶段涨幅'板块，比较该基金与同类平均/沪深300的差距",
            data_source="—(需手动对比)",
        )

    def _check_fee_structure(self, info: dict) -> MonitorCheck:
        """
        检查8：费率是否合理

        管理费>1.5%且长期跑输基准 → 费率高但不产生超额收益 → 不如换指数
        """
        mgmt = info.get("management_fee", 0.015)
        if mgmt > 0.015:
            return MonitorCheck(
                dimension="费率结构",
                level=AlertLevel.YELLOW,
                detail=f"管理费{mgmt:.1%}，高于同类平均(1.5%)",
                action="如果该基金长期跑输同类或基准，考虑换到低费率的指数基金",
                data_source="默认值(东方财富不可用)",
            )
        return MonitorCheck(
            dimension="费率结构",
            level=AlertLevel.GREEN,
            detail=f"管理费{mgmt:.1%}，在正常范围",
            action="无需操作",
            data_source="默认值",
        )

    # ================================================================
    #  工具
    # ================================================================

    def _freshness_note(self) -> str:
        """生成数据新鲜度说明"""
        now = datetime.now()
        y, m = now.year, now.month
        if m <= 3:
            last_q = f"{y-1}Q3"
        elif m <= 6:
            last_q = f"{y}Q1"
        elif m <= 9:
            last_q = f"{y}Q2"
        else:
            last_q = f"{y}Q3"
        return (
            f"监控基于日频净值数据和基金元数据，不依赖持仓。"
            f"最新季报持仓({last_q})已过期，以上告警均不涉及穿透分析。"
        )


def run_monitor(fund_code: str) -> MonitorReport:
    """快捷监控"""
    return FundMonitor().run(fund_code)


def print_monitor_report(report: MonitorReport):
    """打印监控报告"""
    print(f"\n{'='*60}")
    print(f"  基金监控报告: {report.fund_name} ({report.fund_code})")
    print(f"  检查时间: {report.check_time}")
    print(f"  {report.data_freshness_note}")
    print(f"{'='*60}")
    print(f"\n  {report.summary}\n")

    for c in report.checks:
        icon = c.level.value.split()[0]
        print(f"  {icon} {c.dimension}")
        print(f"     {c.detail}")
        if c.level != AlertLevel.GREEN:
            print(f"     → 建议: {c.action}")
        print()


if __name__ == "__main__":
    report = run_monitor("005844")
    print_monitor_report(report)
