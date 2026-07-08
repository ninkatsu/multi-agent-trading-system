"""
Fund Trading Graph - 基金版 LangGraph 编排核心

与股票版 trading_graph.py 的拓扑结构完全一致（框架复用），仅 Agent 替换为基金版。

编排模式：Parallel Fan-out → Fan-in → Sequential Pipeline
    基金代码 → [FundQuality + Technical + Sentiment] (并行)
    → Debate Agent (牛熊辩论)
    → Risk Agent (风控守门, 按基金类型动态阈值)
    → Execution Agent (申购/定投) / Reject (否决)

面试要点：
- 整个编排层零改动！这就是框架的复用价值——拓扑通用，内核替换
- 基金版新增的定投/A-C类/持有期等信号通过 debate_result/risk_assessment 自然流转
- 条件边(conditional_edge) 保持不变：Risk Agent 仍是守门员
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import StateGraph, END

from agents.fund_quality_agent import FundQualityAgent
from agents.technical_agent import FundTechnicalAgent
from agents.sentiment_agent import FundSentimentAgent
from agents.debate_agent import FundDebateAgent
from agents.risk_agent import FundRiskAgent
from agents.execution_agent import FundExecutionAgent


class FundTradingState(TypedDict):
    """基金版全局状态 - 与股票版 TradingState 结构一致"""
    ticker: str  # 复用: 存基金代码
    analyses: Annotated[list[dict[str, Any]], operator.add]
    debate_result: dict[str, Any]
    risk_assessment: dict[str, Any]
    execution_result: dict[str, Any]
    portfolio_value: float


def should_execute(state: FundTradingState) -> str:
    """条件路由：Risk Agent 审批通过 → execute，否则 → reject"""
    risk = state.get("risk_assessment", {})
    if risk.get("approved", False):
        return "execute"
    return "reject"


def reject_node(state: FundTradingState) -> dict[str, Any]:
    """风控否决后的终止节点"""
    risk = state.get("risk_assessment", {})
    return {
        "execution_result": {
            "order_id": "",
            "ticker": state["ticker"],
            "order_type": "none",
            "amount": 0,
            "status": "RISK_REJECTED",
            "message": f"风控否决: {risk.get('reasoning', '')}",
            "violations": risk.get("hard_rule_violations", []),
            "warnings": risk.get("soft_warnings", []),
        }
    }


def build_fund_graph(dry_run: bool = True) -> StateGraph:
    """构建基金投资决策图"""
    fund_quality = FundQualityAgent()
    technical = FundTechnicalAgent()
    sentiment = FundSentimentAgent()
    debate = FundDebateAgent()
    risk = FundRiskAgent()
    execution = FundExecutionAgent(dry_run=dry_run)

    graph = StateGraph(FundTradingState)

    graph.add_node("fund_quality", fund_quality.run)
    graph.add_node("technical", technical.run)
    graph.add_node("sentiment", sentiment.run)
    graph.add_node("debate", debate.run)
    graph.add_node("risk", risk.run)
    graph.add_node("execute", execution.run)
    graph.add_node("reject", reject_node)

    # 并行 Fan-out: START → 三个分析Agent并行
    graph.set_entry_point("fund_quality")
    graph.add_edge("__start__", "technical")
    graph.add_edge("__start__", "sentiment")

    # Fan-in: 三个分析 → debate
    graph.add_edge("fund_quality", "debate")
    graph.add_edge("technical", "debate")
    graph.add_edge("sentiment", "debate")

    # 串行 Pipeline
    graph.add_edge("debate", "risk")

    # 条件路由: 风控守门
    graph.add_conditional_edges(
        "risk",
        should_execute,
        {"execute": "execute", "reject": "reject"},
    )

    graph.add_edge("execute", END)
    graph.add_edge("reject", END)

    return graph


def create_fund_app(dry_run: bool = True):
    """编译基金决策图为可执行应用"""
    graph = build_fund_graph(dry_run=dry_run)
    return graph.compile()


def run_fund_analysis(fund_code: str, portfolio_value: float = 1_000_000,
                      dry_run: bool = True) -> dict[str, Any]:
    """运行完整的基金投资决策流程

    Args:
        fund_code: 基金代码，如 "005827" (易方达蓝筹)
        portfolio_value: 组合总市值
        dry_run: 是否模拟交易

    Returns:
        完整的决策结果状态
    """
    app = create_fund_app(dry_run=dry_run)

    initial_state: FundTradingState = {
        "ticker": fund_code,
        "analyses": [],
        "debate_result": {},
        "risk_assessment": {},
        "execution_result": {},
        "portfolio_value": portfolio_value,
    }

    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("多Agent基金投资决策系统")
    print("=" * 60)

    fund_code = input("请输入基金代码 (默认 005827 易方达蓝筹): ").strip() or "005827"
    print(f"\n正在分析基金 {fund_code}...\n")

    result = run_fund_analysis(fund_code)

    print("\n📊 分析结果:")
    for analysis in result.get("analyses", []):
        agent_name = analysis.get("agent", "unknown")
        print(f"  [{agent_name}] 评分: {analysis['score']}/10 | 信号: {analysis['signal']}")
        print(f"    理由: {analysis['reasoning']}")
        data = analysis.get("data", {})
        if data:
            if agent_name == "fund_quality":
                print(f"    基金经理: {data.get('manager_name', 'N/A')}, "
                      f"穿透PE: {data.get('weighted_pe', 'N/A')}, "
                      f"穿透ROE: {data.get('weighted_roe', 'N/A')}")
                print(f"    4423初筛: {'✅通过' if data.get('pass_4423') else '❌未通过'}")

    debate = result.get("debate_result", {})
    print(f"\n🗣️ 辩论结论: {debate.get('final_signal', 'N/A')} (置信度: {debate.get('confidence', 0):.0%})")
    print(f"  裁决理由: {debate.get('reasoning', 'N/A')}")
    print(f"  建议行动: {debate.get('recommended_action', 'N/A')}")
    print(f"  A/C类建议: {debate.get('a_c_recommendation', 'N/A')} | "
          f"持有期: {debate.get('suggested_holding_period', 'N/A')}")
    watch = debate.get("watch_signals", [])
    if watch:
        print(f"  关注信号: {', '.join(watch)}")

    risk = result.get("risk_assessment", {})
    print(f"\n🛡️ 风控: {'✅ 通过' if risk.get('approved') else '❌ 否决'} "
          f"(基金类型: {risk.get('fund_type', 'N/A')})")
    if risk.get("hard_rule_violations"):
        print(f"  硬规则违规: {risk['hard_rule_violations']}")
    if risk.get("soft_warnings"):
        print(f"  软警告: {risk['soft_warnings']}")

    exec_result = result.get("execution_result", {})
    print(f"\n💰 执行: {exec_result.get('status', 'N/A')}")
    print(f"  {exec_result.get('message', '')}")
    if exec_result.get("dca_plan"):
        print(f"  📅 {exec_result['dca_plan']}")
    if exec_result.get("warnings"):
        for w in exec_result["warnings"]:
            print(f"  {w}")
