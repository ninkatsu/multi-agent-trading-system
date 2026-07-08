#!/usr/bin/env python3
"""
基金多Agent投资决策 — 完整运行入口
用法: python run_fund_analysis.py [基金代码]  (默认: 005844 东方人工智能主题混合A)

输出: 终端打印 + fund_analysis_report.md
"""
from __future__ import annotations

import sys, os, json
from datetime import datetime

# 保证 shared/ 和 fund/ 都在 path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "fund"))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from shared.tools.finance_formulas import (
    safe_div, pct, yoy, cagr, weighted_avg, fund_concentration,
    fund_weighted_valuation, round_safe
)


def main():
    fund_code = sys.argv[1] if len(sys.argv) > 1 else "005844"
    print("=" * 70)
    print(f"  基金多Agent投资决策系统 — {fund_code}")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ====== Step 0: 数据采集 ======
    print("\n【数据采集】正在获取基金数据...")
    from fund.tools.fund_data import get_provider
    provider = get_provider()

    info = provider.get_fund_info(fund_code)
    print(f"  基金名称: {info['fund_name']}")
    print(f"  基金类型: {info['fund_type']}")
    print(f"  费率来源: {info.get('_fee_source', '未知')}")

    nav = provider.get_nav_history(fund_code, period="1y")
    latest_nav = float(nav["unit_nav"].iloc[-1]) if not nav.empty else None
    print(f"  净值数据: {len(nav)}条, 最新净值: {latest_nav}")

    manager = provider.get_fund_manager(fund_code)
    print(f"  基金经理: {manager['name']}, 任期{manager['tenure_years']}年, "
          f"历史回报{manager['historical_return']}%, AUM{manager['aum']}亿")

    look = provider.look_through_analysis(fund_code)
    avail = look.get('_available')
    reason = look.get('_reason', '')
    print(f"  穿透分析: {'可用 [OK]' if avail else '不可用 [WARN] ' + reason}")

    fear = provider.get_market_fear_greed()
    print(f"  市场恐贪: {fear['index']} ({fear['label']})")

    # ====== Step 1: 并行分析 (三个Agent) ======
    print("\n" + "=" * 70)
    print("【Step 1】三Agent并行分析...")
    print("=" * 70)

    # ---- 1a: 基金质量分析 ----
    print("\n--- 1a. 基金质量 (经理+穿透) ---")
    from fund.agents.fund_quality_agent import FundQualityAgent
    quality = FundQualityAgent().analyze(fund_code)
    print(f"  评分: {quality.score}/10, 信号: {quality.signal}")
    print(f"  经理: {quality.manager_name}, 任期{quality.manager_tenure_years}年")
    print(f"  穿透: wPE={quality.weighted_pe}, wROE={quality.weighted_roe}, "
          f"集中度top10={quality.concentration:.1%}")
    print(f"  4423初筛: {'✅通过' if quality.pass_4423_filter else '⚠️未通过'}")
    print(f"  理由: {quality.reasoning}")

    # ---- 1b: 技术面分析 ----
    print("\n--- 1b. 技术面 (净值趋势+定投) ---")
    from fund.agents.technical_agent import FundTechnicalAgent
    tech = FundTechnicalAgent().analyze(fund_code)
    print(f"  评分: {tech.score}/10, 信号: {tech.signal}")
    if tech.current_nav:
        print(f"  当前净值: {tech.current_nav:.4f}, SMA60: {tech.sma_60:.4f}")
        print(f"  偏离SMA60: {tech.nav_deviation_sma60:.1%}" if tech.nav_deviation_sma60 else "  偏离: N/A")
    print(f"  定投评分: {tech.dca_score}/10 → 信号: {tech.dca_signal}")
    print(f"  相对基准: {tech.vs_benchmark_trend}")
    print(f"  理由: {tech.reasoning}")

    # ---- 1c: 情绪面分析 ----
    print("\n--- 1c. 情绪面 (中文NLP) ---")
    from fund.agents.sentiment_agent import FundSentimentAgent
    sent = FundSentimentAgent().analyze(fund_code)
    print(f"  评分: {sent.score}/10, 信号: {sent.signal}")
    print(f"  新闻情绪: {sent.news_sentiment}, 新闻数: {sent.news_count}")
    print(f"  申赎流向: {sent.flow_direction}")
    print(f"  恐贪指数: {sent.fear_greed_index}")
    print(f"  理由: {sent.reasoning}")

    # ====== Step 2: 辩论 ======
    print("\n" + "=" * 70)
    print("【Step 2】牛熊辩论 (2轮) + Judge裁决")
    print("=" * 70)

    analyses = [
        {"agent": "fund_quality", "ticker": fund_code, "score": quality.score,
         "signal": quality.signal, "reasoning": quality.reasoning,
         "data": {"fund_name": quality.fund_name, "manager_name": quality.manager_name,
                  "weighted_pe": quality.weighted_pe, "weighted_roe": quality.weighted_roe}},
        {"agent": "technical", "ticker": fund_code, "score": tech.score,
         "signal": tech.signal, "reasoning": tech.reasoning,
         "data": {"dca_signal": tech.dca_signal, "nav_deviation": tech.nav_deviation_sma60}},
        {"agent": "sentiment", "ticker": fund_code, "score": sent.score,
         "signal": sent.signal, "reasoning": sent.reasoning,
         "data": {"news_sentiment": sent.news_sentiment, "fear_greed": sent.fear_greed_index}},
    ]

    from fund.agents.debate_agent import FundDebateAgent
    debate_result = FundDebateAgent().debate(analyses)

    print(f"\n🐂 Bull论点 ({len(debate_result.bull_arguments)}条):")
    for i, arg in enumerate(debate_result.bull_arguments, 1):
        print(f"  {i}. {arg}")
    print(f"\n🐻 Bear论点 ({len(debate_result.bear_arguments)}条):")
    for i, arg in enumerate(debate_result.bear_arguments, 1):
        print(f"  {i}. {arg}")
    print(f"\n⚖️ Judge裁决:")
    print(f"  信号: {debate_result.final_signal} (置信度: {debate_result.confidence:.0%})")
    print(f"  建议: {debate_result.recommended_action}")
    print(f"  仓位: {debate_result.target_position_pct:.1%}")
    print(f"  A/C类: {debate_result.a_c_recommendation}")
    print(f"  持有期: {debate_result.suggested_holding_period}")
    print(f"  关注信号: {debate_result.watch_signals}")
    print(f"  理由: {debate_result.reasoning}")

    # ====== Step 3: 风控 ======
    print("\n" + "=" * 70)
    print("【Step 3】风控守门")
    print("=" * 70)

    debate_dict = {
        "final_signal": debate_result.final_signal,
        "confidence": debate_result.confidence,
        "reasoning": debate_result.reasoning,
        "target_position_pct": debate_result.target_position_pct,
        "suggested_holding_period": debate_result.suggested_holding_period,
        "recommended_action": debate_result.recommended_action,
    }

    from fund.agents.risk_agent import FundRiskAgent
    risk = FundRiskAgent().assess(fund_code, debate_dict)

    print(f"  基金类型分类: {risk.fund_type}")
    print(f"  审批结果: {'✅ 通过' if risk.approved else '❌ 否决'}")
    if risk.hard_rule_violations:
        print(f"  硬规则违规: {risk.hard_rule_violations}")
    if risk.soft_warnings:
        print(f"  软警告: {risk.soft_warnings}")
    print(f"  调整后仓位: {risk.adjusted_position_pct:.1%}")
    print(f"  持有期建议: {risk.holding_period_advice}")
    print(f"  理由: {risk.reasoning}")

    # ====== Step 4: 执行 ======
    print("\n" + "=" * 70)
    print("【Step 4】执行 (申购/定投)")
    print("=" * 70)

    risk_dict = {
        "approved": risk.approved,
        "adjusted_position_pct": risk.adjusted_position_pct,
        "reasoning": risk.reasoning,
    }

    from fund.agents.execution_agent import FundExecutionAgent
    execution = FundExecutionAgent(dry_run=True).execute(fund_code, risk_dict, debate_dict)

    print(f"  订单ID: {execution.order_id}")
    print(f"  类型: {execution.order_type}")
    print(f"  金额: ¥{execution.amount:,.0f}")
    print(f"  份额类别: {execution.share_class}类")
    print(f"  净值日期: {execution.nav_date}")
    print(f"  申购费: ¥{execution.subscribe_fee:.2f}")
    if execution.estimated_shares:
        print(f"  预计份额: {execution.estimated_shares}")
    if execution.dca_plan:
        print(f"  定投方案: {execution.dca_plan}")
    if execution.warnings:
        for w in execution.warnings:
            print(f"  ⚠️ {w}")
    print(f"  状态: {execution.status}")
    print(f"  信息: {execution.message}")

    # ====== 生成报告 ======
    report_path = os.path.join(PROJECT_ROOT, "fund_analysis_report.md")
    _write_report(report_path, fund_code, info, manager, look, fear, nav,
                  quality, tech, sent, debate_result, risk, execution)
    print(f"\n📄 完整报告已生成: {report_path}")


def _write_report(path, code, info, mgr, look, fear, nav, q, t, s, debate, risk, exe):
    """生成 Markdown 可溯源报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""# 基金投资决策分析报告

> **基金**: {info['fund_name']} ({code})
> **类型**: {info['fund_type']}
> **分析时间**: {now}
> **LLM**: DeepSeek V4 Pro

---

## 一、数据采集

| 项目 | 值 | 来源 |
|------|-----|------|
| 基金名称 | {info['fund_name']} | akshare fund_name_em |
| 基金类型 | {info['fund_type']} | akshare fund_name_em |
| 管理费率 | {info['management_fee']:.1%}/年 | {info.get('_fee_source', '默认值')} |
| 托管费率 | {info['custody_fee']:.2%}/年 | 同上 |
| 申购费率(最高) | {info['subscribe_fee']:.1%} | 同上 |
| 基金经理 | {mgr['name']} | akshare fund_manager_em |
| 任职天数 | {mgr['tenure_days']}天 (~{mgr['tenure_years']}年) | akshare fund_manager_em |
| 历史最佳回报 | {mgr['historical_return']}% | akshare fund_manager_em |
| 管理AUM | {mgr['aum']}亿 | akshare fund_manager_em |
| 最新净值 (NAV) | {float(nav['unit_nav'].iloc[-1]):.4f} | akshare fund_open_fund_info_em |
| 净值数据量 | {len(nav)}条 | akshare fund_open_fund_info_em |

> **数据可用性说明**: 持仓数据(季报)当前不可用 → 穿透分析跳过, 评分中以基金经理评估为主.

---

## 二、三Agent并行分析

### 2.1 基金质量分析 (FundQualityAgent)

| 指标 | 值 | 计算方式 |
|------|-----|----------|
| 综合评分 | {q.score}/10 | LLM多维度综合 |
| 信号 | {q.signal} | BUY/HOLD/SELL |
| 经理任期 | {q.manager_tenure_years}年 | akshare 累计从业时间 |
| 穿透PE | {q.weighted_pe or 'N/A'} | Σ(PE_i × weight_i) / Σ(weight_i) — 需持仓数据 |
| 穿透ROE | {q.weighted_roe or 'N/A'} | Σ(ROE_i × weight_i) / Σ(weight_i) |
| 持仓集中度(top10) | {q.concentration:.1%} | Σ(top10 weights) |
| 4423初筛 | {'✅通过' if q.pass_4423_filter else '⚠️未通过'} | 近1年同类前1/4 → 近2年同类前1/4 → 近3年同类前1/2 → 近5年同类前1/3 |

**分析理由**: {q.reasoning}

### 2.2 技术面分析 (TechnicalAgent)

| 指标 | 值 | 公式 |
|------|-----|------|
| 综合评分 | {t.score}/10 | LLM多维度综合 |
| 当前净值 | {t.current_nav:.4f} | akshare 单位净值走势 |
| SMA(60) | {t.sma_60:.4f} | 60日简单移动平均 |
| 偏离SMA60 | {t.nav_deviation_sma60:.1%} | (NAV - SMA60) / SMA60 |
| 定投评分 | {t.dca_score}/10 | 偏离度+回撤+横盘判断 |
| 定投信号 | {t.dca_signal} | START/INCREASE/KEEP/REDUCE |

**分析理由**: {t.reasoning}

### 2.3 情绪面分析 (SentimentAgent)

| 指标 | 值 | 来源 |
|------|-----|------|
| 综合评分 | {s.score}/10 | LLM中文情绪判断 |
| 新闻情绪 | {s.news_sentiment} (-1~+1) | LLM zero-shot 中文NLP |
| 新闻数量 | {s.news_count}条 | 东方财富新闻 |
| 市场恐贪 | {fear['index']:.0f} ({fear['label']}) | 沪深300偏离SMA60映射 |

**分析理由**: {s.reasoning}

---

## 三、辩论 (DebateAgent — 2轮)

### Bull (看多) 论点
""")
        for i, arg in enumerate(debate.bull_arguments, 1):
            f.write(f"{i}. {arg}\n")

        f.write(f"""
### Bear (看空) 论点
""")
        for i, arg in enumerate(debate.bear_arguments, 1):
            f.write(f"{i}. {arg}\n")

        f.write(f"""
### Judge 裁决

| 项目 | 值 |
|------|-----|
| 最终信号 | **{debate.final_signal}** |
| 置信度 | {debate.confidence:.0%} |
| 建议仓位 | {debate.target_position_pct:.1%} |
| A/C类建议 | {debate.a_c_recommendation} |
| 建议持有期 | {debate.suggested_holding_period} |
| 关注信号 | {', '.join(debate.watch_signals)} |

**裁决理由**: {debate.reasoning}

**建议操作**: {debate.recommended_action}

---

## 四、风控 (RiskAgent)

| 检查项 | 结果 |
|--------|------|
| 基金风控类型 | {risk.fund_type} |
| 审批 | {'✅ 通过' if risk.approved else '❌ 否决'} |
| 仓位上限 | {risk.max_position_allowed:.0%} |
| 调整后仓位 | {risk.adjusted_position_pct:.1%} |
""")
        if risk.hard_rule_violations:
            f.write(f"| 硬规则违规 | {', '.join(risk.hard_rule_violations)} |\n")
        if risk.soft_warnings:
            f.write(f"| 软警告 | {', '.join(risk.soft_warnings)} |\n")

        f.write(f"""
**风控理由**: {risk.reasoning}

---

## 五、执行 (ExecutionAgent)

| 项目 | 值 |
|------|-----|
| 订单ID | {exe.order_id} |
| 操作类型 | {exe.order_type} |
| 金额 | ¥{exe.amount:,.0f} |
| 份额类别 | {exe.share_class}类 |
| 净值日期 | {exe.nav_date} |
| 申购费 | ¥{exe.subscribe_fee:.2f} |
""")
        if exe.estimated_shares:
            f.write(f"| 预计份额 | {exe.estimated_shares} |\n")
        if exe.dca_plan:
            f.write(f"| 定投计划 | {exe.dca_plan} |\n")
        f.write(f"""
**状态**: {exe.status}
**信息**: {exe.message}
""")
        if exe.warnings:
            f.write("\n### ⚠️ 风险提示\n\n")
            for w in exe.warnings:
                f.write(f"- {w}\n")

        f.write(f"""
---

## 六、溯源索引

| 决策步骤 | 输入数据 | 计算/推理 | 输出 |
|----------|---------|----------|------|
| 基金质量 | fund_name_em + fund_manager_em + NAV | LLM多维度评分 | {q.score}/10 → {q.signal} |
| 技术面 | NAV日频 × {len(nav)}条 | SMA20/SMA60 + 偏离度 | {t.score}/10 → {t.signal} |
| 情绪面 | 新闻 × {s.news_count}条 | LLM zero-shot中文NLP | {s.score}/10 → {s.signal} |
| 辩论 | 3个分析结果 | 2轮Bull↔Bear对抗 + Judge裁决 | {debate.final_signal} (置信度{debate.confidence:.0%}) |
| 风控 | 辩论结果 + 基金信息 | 硬规则(阈值{risk.max_position_allowed:.0%}) + LLM软判断 | {'通过' if risk.approved else '否决'} |
| 执行 | 风控结果 + 费率 | 15:00时间窗 + A/C选择 + 费率计算 | 申购 ¥{exe.amount:,.0f} ({exe.share_class}类) |

---

> ⚠️ **免责声明**: 本报告由AI自动生成, 仅供学习参考, 不构成任何投资建议。
> 数据源: akshare(免费), yfinance(免费), DeepSeek API.
> 生成时间: {now}
""")

    print(f"  Report written to: {path}")


if __name__ == "__main__":
    main()
