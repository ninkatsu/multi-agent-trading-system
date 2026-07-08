"""
Fund Sentiment Agent - 基金情绪面分析Agent

职责：中文新闻情绪分析、基金吧/社区热度、申赎资金流向、晨星评级。

与股票版的核心区别：
- TextBlob 对中文无效！直接用 LLM 做中文 zero-shot 情绪判断
- 数据源从英文新闻变为中文新闻 + 基金吧讨论
- 机构持仓 → 申赎资金流向（净申购=受欢迎）
- 分析师评级 → 晨星星级/天天基金评分
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.llm import get_llm
from config.json_utils import parse_json_loose
from config.settings import CONFIG
from tools.fund_data import get_provider


@dataclass
class FundSentimentAnalysis:
    fund_code: str
    news_sentiment: float  # -1 到 +1
    news_count: int
    fund_bar_sentiment: float | None  # 基金吧情绪
    flow_direction: str  # 申赎流向
    morningstar_stars: int
    tiantian_score: float
    fear_greed_index: float  # 整体市场恐贪
    score: float
    signal: str
    reasoning: str


class FundSentimentAgent:
    """基金情绪面分析：中文新闻情绪(LLM) + 基金吧热度 + 申赎流向 + 评级"""

    SYSTEM_PROMPT = """你是一位市场情绪分析专家，专注国内公募基金。请基于以下数据评估市场对该基金的情绪状态。

评分标准 (1-10分):
- 新闻情绪: 偏正面(+2), 中性(0), 偏负面(-2)
- 申赎资金: 净申购(+2), 持平(0), 净赎回(-2)
- 社区热度: 关注度上升+正面(0~+1), 关注度低(0), 骂声一片(-1)
- 评级: 晨星★★★★以上(+1), ★★以下(-1)
- 市场恐贪: 恐惧(<30)反是机会(+1), 极度贪婪(>80)谨慎(-1)
- 基金吧讨论质量: 理性讨论(+1), 情绪化吐槽(-1)

请输出JSON:
{
    "score": <1-10>,
    "signal": "<BUY/SELL/HOLD>",
    "reasoning": "<100字分析>",
    "sentiment_summary": "<一句话总结市场情绪>"
}"""

    CHINESE_SENTIMENT_PROMPT = """请分析以下中文新闻标题的情绪倾向。

新闻列表:
{news_titles}

请判断:
- 每条新闻的情绪: 正面(1) / 中性(0) / 负面(-1)
- 综合所有新闻，输出:
  - overall_sentiment: -1到+1 的综合情绪分
  - key_themes: 大家都在讨论什么(3个关键词或主题)
  - summary: 一句话总结

输出JSON: {{"overall_sentiment": 0.0, "key_themes": ["主题1", "主题2", "主题3"], "summary": "..."}}"""

    def __init__(self):
        self.llm = get_llm(temperature=CONFIG.llm.temperature)
        self.provider = get_provider()

    def _analyze_chinese_news_sentiment(self, fund_code: str) -> tuple[float, int, str]:
        """用LLM直接判断中文新闻情绪（不依赖TextBlob）"""
        news = self.provider.get_fund_news(fund_code, limit=15)
        if not news:
            return 0.0, 0, "无新闻数据"

        titles = [n.get("title", "") for n in news if n.get("title")]
        if not titles:
            return 0.0, 0, "无标题可分析"

        titles_text = "\n".join(f"- {t}" for t in titles[:15])
        prompt = self.CHINESE_SENTIMENT_PROMPT.format(news_titles=titles_text)

        try:
            response = self.llm.invoke([
                SystemMessage(content="你是一个中文金融新闻情绪分析工具。只输出JSON，不要额外解释。"),
                HumanMessage(content=prompt),
            ])
            result = parse_json_loose(response.content) or {}
            sentiment = float(result.get("overall_sentiment", 0.0))
            summary = str(result.get("summary", ""))
            return max(-1.0, min(1.0, sentiment)), len(titles), summary
        except Exception:
            return 0.0, len(titles), "情绪分析出错"

    def _get_fund_flow_direction(self, fund_code: str) -> str:
        """判断申赎资金流向"""
        # akshare 有资金流向接口，这里做简化
        try:
            import akshare as ak
            # 尝试获取基金规模变化来推断资金流向
            info = self.provider.get_fund_info(fund_code)
            size = info.get("fund_size", 0)
            if size > 100:
                return "NET_INFLOW_LARGE"  # 大规模基金，假定持续有资金
            elif size > 10:
                return "NET_INFLOW"
            return "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def analyze(self, fund_code: str) -> FundSentimentAnalysis:
        news_sentiment, news_count, news_summary = self._analyze_chinese_news_sentiment(fund_code)
        flow = self._get_fund_flow_direction(fund_code)
        rating = self.provider.get_fund_rating(fund_code)
        fear_greed = self.provider.get_market_fear_greed()
        info = self.provider.get_fund_info(fund_code)

        user_prompt = f"""请分析基金 {fund_code} ({info.get('fund_name', '')}) 的市场情绪：

【新闻情绪】
- LLM判断综合情绪: {news_sentiment} (范围-1~+1)
- 新闻数量: {news_count} 条
- 摘要: {news_summary}

【申赎资金流向】{flow}

【评级】
- 晨星星级: {rating.get('morningstar_stars', 'N/A')}
- 天天基金评分: {rating.get('tiantian_score', 'N/A')}

【市场恐贪指数】{fear_greed.get('index', 50):.0f} - {fear_greed.get('label', '未知')}"""

        response = self.llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        result = parse_json_loose(response.content) or {
            "score": 5.0, "signal": "HOLD", "reasoning": "LLM输出解析失败",
            "sentiment_summary": "",
        }

        return FundSentimentAnalysis(
            fund_code=fund_code,
            news_sentiment=news_sentiment,
            news_count=news_count,
            fund_bar_sentiment=None,
            flow_direction=flow,
            morningstar_stars=int(rating.get("morningstar_stars", 0)),
            tiantian_score=float(rating.get("tiantian_score", 0)),
            fear_greed_index=float(fear_greed.get("index", 50)),
            score=float(result.get("score", 5.0)),
            signal=str(result.get("signal", "HOLD")),
            reasoning=str(result.get("reasoning", "")),
        )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        fund_code = state["ticker"]
        analysis = self.analyze(fund_code)
        return {
            "analyses": [{
                "agent": "sentiment",
                "ticker": fund_code,
                "score": analysis.score,
                "signal": analysis.signal,
                "reasoning": analysis.reasoning,
                "data": {
                    "news_sentiment": analysis.news_sentiment,
                    "news_count": analysis.news_count,
                    "flow_direction": analysis.flow_direction,
                    "morningstar_stars": analysis.morningstar_stars,
                    "fear_greed_index": analysis.fear_greed_index,
                },
            }]
        }
