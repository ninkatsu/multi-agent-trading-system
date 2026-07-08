#!/usr/bin/env python3
"""
基金多Agent投资决策系统 — 三模式统一入口

用法:
  python run_fund_analysis.py quarterly 005844   季度深度分析（季报公布后1-2周内运行）
  python run_fund_analysis.py monitor 005844     日常监控告警（每周运行）
  python run_fund_analysis.py dca 005844 [金额]  定投时点评估（每周运行，可选参数：每周定投额）
  python run_fund_analysis.py 005844             自动模式（根据数据新鲜度自动选择）

三模式定位:
  quarterly → 一年4次，持仓新鲜时做最深度的全Agent分析 → 输出"持有/加仓/减仓/换基"
  monitor  → 每周1次，不依赖持仓的硬信号检查 → 输出"有没有危险"
  dca      → 每周1次，净值位置+情绪判断 → 输出"这周投多少"
"""
from __future__ import annotations

import sys, os, json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "fund"))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from fund.tools.fund_data import get_provider
from shared.tools.finance_formulas import safe_div, pct, yoy, cagr


def main():
    # 解析参数
    args = sys.argv[1:]
    if not args:
        print_usage()
        return

    if len(args) >= 2 and args[0] in ("quarterly", "monitor", "dca"):
        mode = args[0]
        fund_code = args[1]
        dca_amount = float(args[2]) if mode == "dca" and len(args) >= 3 else 2000
    else:
        # 自动模式：只给基金代码
        fund_code = args[0]
        mode = auto_detect_mode(fund_code)
        dca_amount = 2000
        print(f"[自动模式] 检测到数据新鲜度适合: {mode}模式")

    # 分发
    if mode == "quarterly":
        run_quarterly_deep_analysis(fund_code)
    elif mode == "monitor":
        run_monitor_mode(fund_code)
    elif mode == "dca":
        run_dca_mode(fund_code, dca_amount)


def auto_detect_mode(fund_code: str) -> str:
    """
    根据距最近季报的天数自动选择模式

    规则：
      距季报截止日 < 21天  → quarterly（数据新鲜，值得深度分析）
      距季报截止日 21-60天 → monitor（数据开始变旧，监控为主）
      距季报截止日 > 60天  → monitor（太旧了，深度分析无意义）
    但每周都可以跑 dca
    """
    now = datetime.now()
    y, m = now.year, now.month

    # 最近季报截止日
    if m <= 3:
        last_report = datetime(y-1, 9, 30)
    elif m <= 6:
        last_report = datetime(y, 3, 31)
    elif m <= 9:
        last_report = datetime(y, 6, 30)
    else:
        last_report = datetime(y, 9, 30)

    days_since_report = (now - last_report).days

    if days_since_report < 21:
        return "quarterly"
    else:
        return "monitor"


def print_usage():
    print("""
基金多Agent投资决策系统 — 三模式

用法:
  python run_fund_analysis.py quarterly <基金代码>  季度深度分析
  python run_fund_analysis.py monitor <基金代码>    日常监控告警
  python run_fund_analysis.py dca <基金代码> [金额]  定投时点评估
  python run_fund_analysis.py <基金代码>             自动选择模式

示例:
  python run_fund_analysis.py quarterly 005844
  python run_fund_analysis.py monitor 005844
  python run_fund_analysis.py dca 005844 3000
""")


# ================================================================
#  模式1: 季度深度分析（完整6Agent链路）
# ================================================================

def run_quarterly_deep_analysis(fund_code: str):
    """季报公布后的深度分析 — 全Agent链路"""
    from fund.tools.style_analysis import run_style_analysis

    print_header("季度深度分析", fund_code)

    # 数据采集
    print("\n【数据采集】")
    provider = get_provider()
    info = provider.get_fund_info(fund_code)
    nav = provider.get_nav_history(fund_code, period="1y")
    mgr = provider.get_fund_manager(fund_code)
    look = provider.look_through_analysis(fund_code)
    style = run_style_analysis(fund_code)
    fear = provider.get_market_fear_greed()

    print(f"  基金: {info['fund_name']} | 经理: {mgr['name']} | AUM: {mgr['aum']}亿")
    print(f"  NAV: {float(nav['unit_nav'].iloc[-1]):.4f} | 数据: {len(nav)}条")
    print(f"  穿透: {'可用' if look.get('_available') else '不可用'} | "
          f"风格: {style.method}(置信度:{style.confidence})")

    # 数据新鲜度警告
    print(f"\n  ⚠️ 持仓黑箱期: {style.blackout_warning.split(chr(10))[1]}")

    # 三Agent分析
    print("\n【Step 1】三Agent并行分析")
    from fund.agents.fund_quality_agent import FundQualityAgent
    from fund.agents.technical_agent import FundTechnicalAgent
    from fund.agents.sentiment_agent import FundSentimentAgent

    quality = FundQualityAgent().analyze(fund_code)
    tech = FundTechnicalAgent().analyze(fund_code)
    sent = FundSentimentAgent().analyze(fund_code)

    for label, a in [("基金质量", quality), ("技术面", tech), ("情绪面", sent)]:
        print(f"  {label}: {a.score}/10 → {a.signal} | {a.reasoning[:80]}...")

    # 辩论
    print("\n【Step 2】牛熊辩论")
    analyses = [
        {"agent": "fund_quality", "ticker": fund_code, "score": quality.score,
         "signal": quality.signal, "reasoning": quality.reasoning,
         "data": {"fund_name": quality.fund_name, "manager_name": quality.manager_name}},
        {"agent": "technical", "ticker": fund_code, "score": tech.score,
         "signal": tech.signal, "reasoning": tech.reasoning,
         "data": {"dca_signal": tech.dca_signal, "nav_deviation": tech.nav_deviation_sma60}},
        {"agent": "sentiment", "ticker": fund_code, "score": sent.score,
         "signal": sent.signal, "reasoning": sent.reasoning,
         "data": {"news_sentiment": sent.news_sentiment, "fear_greed": sent.fear_greed_index}},
    ]

    from fund.agents.debate_agent import FundDebateAgent
    debate = FundDebateAgent().debate(analyses)

    print(f"  Bull({len(debate.bull_arguments)}条) vs Bear({len(debate.bear_arguments)}条)")
    print(f"  裁决: {debate.final_signal} (置信度{debate.confidence:.0%})")
    print(f"  建议: {debate.recommended_action}")

    # 风控
    print("\n【Step 3】风控")
    from fund.agents.risk_agent import FundRiskAgent
    risk = FundRiskAgent().assess(fund_code, {
        "final_signal": debate.final_signal, "confidence": debate.confidence,
        "target_position_pct": debate.target_position_pct,
        "suggested_holding_period": debate.suggested_holding_period,
    })
    print(f"  {'✅通过' if risk.approved else '❌否决'} | 仓位{risk.adjusted_position_pct:.1%} | {risk.reasoning[:80]}...")

    # 执行
    print("\n【Step 4】执行")
    from fund.agents.execution_agent import FundExecutionAgent
    exe = FundExecutionAgent(dry_run=True).execute(
        fund_code,
        {"approved": risk.approved, "adjusted_position_pct": risk.adjusted_position_pct},
        {"final_signal": debate.final_signal, "suggested_holding_period": debate.suggested_holding_period,
         "recommended_action": debate.recommended_action, "a_c_recommendation": debate.a_c_recommendation},
    )
    print(f"  {exe.message}")

    # 生成报告
    report_path = os.path.join(PROJECT_ROOT, f"fund_quarterly_report_{fund_code}.md")
    _write_quarterly_report(report_path, fund_code, info, mgr, look, style, nav,
                            quality, tech, sent, debate, risk, exe, fear)
    print(f"\n📄 季度深度报告: {report_path}")


# ================================================================
#  模式2: 日常监控告警
# ================================================================

def run_monitor_mode(fund_code: str):
    """独立告警信号检查 — 不调LLM"""
    from fund.tools.monitor import run_monitor, print_monitor_report
    report = run_monitor(fund_code)
    print_monitor_report(report)

    # 同时跑一个轻量的定投信号
    from fund.tools.dca_assistant import assess_dca, print_dca_signal
    dca = assess_dca(fund_code)
    print_dca_signal(dca)


# ================================================================
#  模式3: 定投时点评估
# ================================================================

def run_dca_mode(fund_code: str, weekly_amount: float = 2000):
    """定投信号 — 纯公式，不调LLM"""
    from fund.tools.dca_assistant import assess_dca, print_dca_signal
    signal = assess_dca(fund_code, weekly_amount)
    print_dca_signal(signal)

    # 同时跑监控检查（免费，不增加API成本）
    from fund.tools.monitor import run_monitor, print_monitor_report
    report = run_monitor(fund_code)
    if report.has_critical_alert:
        print("⚠️ 监测到红色警报，定投可能需要暂停！详见下方：")
        for c in report.checks:
            if c.level.value.startswith("🔴"):
                print(f"  🔴 {c.dimension}: {c.detail}")
                print(f"     → {c.action}")


# ================================================================
#  报告生成
# ================================================================

def print_header(mode: str, fund_code: str):
    print(f"\n{'='*70}")
    print(f"  {mode}: {fund_code}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")


def _write_quarterly_report(path, code, info, mgr, look, style, nav,
                            q, t, s, debate, risk, exe, fear):
    """生成季度深度分析 Markdown 报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    nav_last = float(nav["unit_nav"].iloc[-1]) if not nav.empty and "unit_nav" in nav.columns else 0

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""# 基金季度深度分析报告

> **基金**: {info['fund_name']} ({code})
> **类型**: {info['fund_type']}
> **分析时间**: {now}
> **LLM**: DeepSeek V4 Pro
> **模式**: 季度深度分析（适合季报公布后1-2周内运行）

---

## ⚠️ 持仓黑箱期声明

{style.blackout_warning}

---

## 一、基金概况

| 项目 | 值 | 来源 |
|------|-----|------|
| 基金名称 | {info['fund_name']} | akshare |
| 基金经理 | {mgr['name']} | akshare fund_manager_em |
| 任职天数 | {mgr['tenure_days']}天 (~{mgr['tenure_years']}年) | 同上 |
| 管理AUM | {mgr['aum']}亿 | 同上 |
| 最新净值 | {nav_last:.4f} | akshare 日频 |
| 管理费率 | {info['management_fee']:.1%}/年 | {info.get('_fee_source', '默认值')} |

---

## 二、六Agent决策链

### 2.1 FundQualityAgent — 基金质量

评分: **{q.score}/10 → {q.signal}**

分析理由: {q.reasoning}

穿透持仓指标:
- 加权PE: {q.weighted_pe or 'N/A'}
- 加权ROE: {q.weighted_roe or 'N/A'}
- 集中度(top10): {q.concentration:.1%}
- 4423初筛: {'通过' if q.pass_4423_filter else '未通过'}

### 2.2 TechnicalAgent — 技术面

评分: **{t.score}/10 → {t.signal}**

| 指标 | 值 | 公式 |
|------|-----|------|
| 当前净值 | {t.current_nav:.4f} | — |
| SMA(60) | {t.sma_60:.4f} | 60日简单移动平均 |
| 偏离SMA60 | {t.nav_deviation_sma60:.1%} if t.nav_deviation_sma60 else 'N/A' | (NAV-SMA60)/SMA60 |
| 定投信号 | {t.dca_signal} | 偏离度+回撤综合评分 |

### 2.3 SentimentAgent — 情绪面

评分: **{s.score}/10 → {s.signal}**

| 指标 | 值 |
|------|-----|
| 新闻情绪 | {s.news_sentiment} |
| 市场恐贪 | {fear['index']:.0f} ({fear['label']}) |
| 申赎流向 | {s.flow_direction} |

### 2.4 DebateAgent — 牛熊辩论

**裁决: {debate.final_signal} (置信度 {debate.confidence:.0%})**

Bull论点:
""")
        for i, arg in enumerate(debate.bull_arguments, 1):
            f.write(f"{i}. {arg}\n")

        f.write(f"""
Bear论点:
""")
        for i, arg in enumerate(debate.bear_arguments, 1):
            f.write(f"{i}. {arg}\n")

        f.write(f"""
裁决理由: {debate.reasoning}

建议操作: {debate.recommended_action}
A/C类建议: {debate.a_c_recommendation} | 持有期: {debate.suggested_holding_period}
关注信号: {', '.join(debate.watch_signals)}

### 2.5 RiskAgent — 风控

| 项目 | 值 |
|------|-----|
| 基金类型 | {risk.fund_type} |
| 审批 | {'✅ 通过' if risk.approved else '❌ 否决'} |
| 调整后仓位 | {risk.adjusted_position_pct:.1%} |
""")
        if risk.hard_rule_violations:
            f.write(f"| 硬规则违规 | {', '.join(risk.hard_rule_violations)} |\n")
        if risk.soft_warnings:
            f.write(f"| 软警告 | {', '.join(risk.soft_warnings)} |\n")

        f.write(f"""
### 2.6 ExecutionAgent — 执行

| 项目 | 值 |
|------|-----|
| 类型 | {exe.order_type} |
| 金额 | ¥{exe.amount:,.0f} |
| 份额类别 | {exe.share_class}类 |
""")
        if exe.dca_plan:
            f.write(f"| 定投计划 | {exe.dca_plan} |\n")
        if exe.warnings:
            f.write("\n### ⚠️ 风险提示\n")
            for w in exe.warnings:
                f.write(f"- {w}\n")

        f.write(f"""
---

## 三、溯源索引

| 步骤 | 输入 | 方法 | 输出 |
|------|------|------|------|
| 基金质量 | fund_manager_em + NAV | LLM多维度评分 | {q.score}/10 → {q.signal} |
| 技术面 | NAV × {len(nav)}条 | SMA + 偏离度 | {t.score}/10 → {t.signal} |
| 情绪面 | 新闻 + 资金流 | LLM中文NLP | {s.score}/10 → {s.signal} |
| 辩论 | 3个分析结果 | 2轮对抗+Judge | {debate.final_signal}({debate.confidence:.0%}) |
| 风控 | 辩论+基金信息 | 类型阈值+LLM | {'通过' if risk.approved else '否决'} |
| 执行 | 风控+费率 | 申赎/定投 | {exe.status} |

---

> ⚠️ 本报告由AI生成，仅供学习参考，不构成投资建议。
> 生成时间: {now}
""")


if __name__ == "__main__":
    main()
