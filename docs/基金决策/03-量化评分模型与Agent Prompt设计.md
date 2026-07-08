# 基金决策 03 · 量化评分模型与 Agent Prompt 设计

> 本文是连接"金融理论"和"代码实现"的桥梁。每个 Agent 的评分公式、权重、Prompt 模板都在这里——读完可以直接写代码。

---

## 一、基金基本面 Agent (FundQualityAgent) 评分模型

### 1.1 评分维度与权重

基金基本面不同于股票 PE/PB，需要多维度综合评估。以下评分体系参考晨星（Morningstar）方法论 + A股市场特点：

```
总分 = Σ(维度分 × 权重) × 类型调整系数

维度          权重    满分    数据源
────────────────────────────────────
基金经理        30%    10      akshare 基金经理档案
持仓质量        25%    10      穿透分析（加权PE/ROE等）
历史业绩        20%    10      净值历史 + 基准对比
基金公司        10%    10      管理规模/团队稳定性
规模与流动性    10%    10      规模 + 换手率
费率结构         5%    10      综合费率
```

### 1.2 各维度评分细则

#### 基金经理 (30分)

```
评分标准:
  - 任职本基金年限: >5年(+3), 3-5年(+2), 1-3年(+1), <1年(0)
  - 历史年化收益(任职以来): >15%(+3), 10-15%(+2), 5-10%(+1), <5%或亏损(0)
  - 超额收益(vs 基准): >5%(+2), 0-5%(+1), 跑输(0)
  - 历史最大回撤控制: <基准回撤(+1), ≈基准(0), >基准(-1)
  - 业绩稳定性(年度分位): 连续3年同类前50%(+1), 大起大落(0)

总分换算: 原始分/10 × 10 → 0-10分
```

#### 持仓质量 (25分) — 穿透分析 ⭐

这是你提出的"加权持仓分析"——基金版独有的核心创新：

```python
# 穿透分析算法伪代码
def look_through_analysis(fund_code):
    holdings = get_fund_holdings(fund_code)  # [{stock: "600519", weight: 0.098}, ...]
    
    weighted_pe = 0
    weighted_roe = 0
    weighted_pb = 0
    concentration = 0  # 前10大重仓占比
    
    for h in holdings:
        stock_financials = get_stock_fundamentals(h.stock)
        weighted_pe += stock_financials.pe * h.weight
        weighted_roe += stock_financials.roe * h.weight
        weighted_pb += stock_financials.pb * h.weight
    
    top10_sum = sum(h.weight for h in holdings[:10])
    
    return {
        "weighted_pe": weighted_pe,        # 加权市盈率
        "weighted_roe": weighted_roe,      # 加权ROE
        "weighted_pb": weighted_pb,        # 加权市净率
        "concentration": top10_sum,        # 集中度
        "stock_count": len(holdings),      # 持仓数量
    }
```

```
评分标准:
  - 加权ROE: >18%(+3), 12-18%(+2), 8-12%(+1), <8%(0)
  - 加权PE合理性: <15(+2), 15-25(+1), >30(-1, 偏贵)
  - 集中度(前10): 40-60%适中(+2), 30-40%(+1), >80%太集中(-1), <20%太散(0)
  - 行业分散度: 跨≥5个行业(+2), 3-4个(+1), ≤2个集中(-1)
  - 重仓股质量: 重仓股中有≥3只公认白马(+1)

总分换算至0-10
```

#### 历史业绩 (20分)

```
评分标准:
  - 近1年收益率 vs 同类平均: 前25%(+3), 25-50%(+2), 50-75%(+1), 后25%(-1)
  - 近3年年化 vs 同类: 前25%(+3), 25-50%(+2), 50-75%(+1), 后25%(-1)
  - 夏普比率: >1.5(+2), 1.0-1.5(+1), <0.5(0)
  - 最大回撤 vs 同类: 小于平均(+2), 接近平均(+1), 大于平均(-1)
```

#### 基金公司 (10分)

```
评分标准:
  - 管理规模排名: 前10(+3), 前20(+2), 前50(+1)
  - 投研团队人数: >50人(+1)
  - 公司旗下基金整体表现: 多数同类前50%(+2)
  - 是否有"双十"基金经理(管10年以上/年化>10%): (+1)
  - 近3年有无重大负面(老鼠仓/踩雷): 无(0), 有(-3)
```

#### 规模与流动性 (10分)

```
评分标准:
  - 规模: 10-100亿(+3), 1-10亿或100-300亿(+2), <1亿(清盘风险-2)或>500亿(+0, 难操作)
  - 换手率(年度): 100-300%适中(+2), <100%或300-500%(+1), >500%高频(-1)
  - 申赎状态: 开放申购(+2), 限制大额(+1), 暂停申购(-2)
```

#### 费率结构 (5分)

```
评分标准:
  - 管理费 vs 同类: 低于平均(+2), 等于平均(+1), 高于平均(0)
  - 托管费: 低于0.2%(+1)
  - 综合费率(管理+托管+销售): <1.0%(+2), 1.0-1.5%(+1), >2.0%(0)
```

### 1.3 类型调整系数

不同类型的基金基本面评估标准不同，用系数修正：

```
基金类型            调整系数   说明
股票型/偏股混合      ×1.0      标准评分
平衡混合型           ×0.9      更看重稳健而非高收益
偏债混合型           ×0.8      债券部分基本面评估不同
纯债基金             ×0.7      用债券专属评分体系
指数基金             ×0.5      不看经理选股，看跟踪误差
QDII                 ×0.8      海外资产，信息不对称
```

### 1.4 4423 法则快速筛选

这是国内基金圈的经典初筛方法（可以集成到 Agent 的第一步）：

```
4: 近1年同类排名前1/4
4: 近2年同类排名前1/4
2: 近3年同类排名前1/2
3: 近5年同类排名前1/3

+ 基金经理任职 >3年
+ 规模 >2亿(防清盘)

满足以上条件 → 才进入详细6维度评分
不满足 → 直接标记"初筛不通过，建议观察"
```

---

## 二、基金技术面 Agent (TechnicalAgent) 评分模型

### 2.1 净值趋势评分

普通基金只有日频净值，没有盘中K线。指标体系与股票版不同：

```
指标              买入信号                   卖出信号                  ETF适用
均线多头排列      净值>SMA20>SMA60           净值<SMA20<SMA60          ✅
SMA偏离           净值偏离SMA60 <5%(低位)     净值偏离SMA60 >20%(高位) ✅
净值创新高        突破前20日高点(+2)          跌穿前20日低点(-2)       ✅
相对强弱 vs 基准   跑赢基准且趋势向上(+2)      跑输基准且加速(-2)       ✅
成交量(ETF only)   放量上涨(+1)               放量下跌(-1)             ✅ETF only
波动率收缩         布林带收窄至6月低点(+1)     布林带急剧扩张(-1)       ✅

普通基金总分: 8分制(无成交量指标) ÷ 8 × 10 → 0-10
ETF总分: 10分制 ÷ 10 × 10 → 0-10
```

### 2.2 定投择时信号（基金版独有）

基金版独有的"是否适合开始定投"信号：

```
定投启动信号(满分10):
  - 净值低于SMA60(低位区域): +3
  - 基金回撤>10% vs 近一年高点: +3
  - 市场恐贪指数<30(恐惧): +2
  - 申赎资金净流出但速度放缓(底部信号): +1
  - 净值在SMA20附近横盘>10天(筑底): +1

总分>6 → "建议启动/加码定投"
总分4-6 → "维持定投"
总分<4 → "暂停定投，等待更好时机"
```

---

## 三、基金情绪面 Agent (SentimentAgent) 评分模型

### 3.1 中文情绪分析方案

TextBlob 对中文无效 → 直接用 GLM 做 zero-shot 情绪判断：

```python
# 不再用 TextBlob，而是用 LLM 直接判中文情绪
CHINESE_SENTIMENT_PROMPT = """请分析以下基金相关新闻标题的情绪倾向：

新闻列表:
{news_titles}

对每条新闻，判断:
- sentiment: 正面(1) / 中性(0) / 负面(-1)
- intensity: 情绪强度(0-1, 越极端越接近1)
- reason: 1句话说明

最后输出综合情绪得分(-1到+1)和一段50字的分析。
输出JSON格式: {"overall_sentiment": 0.3, "positive_count": 3, ...}
"""
```

### 3.2 基金情绪指标体系

```
指标                  权重    正面                        负面
新闻情绪(GLM中文判)     30%    利好政策/业绩优秀/资金流入    踩雷/经理离职/限购/巨额赎回
申赎资金流向            25%    连续净申购                    持续净赎回
基金吧/社区热度         15%    讨论活跃+正面占比>60%         骂声一片/冷清
晨星/天天基金评级       15%    ★★★★以上                    ★★以下
机构调研动向            10%    机构密集调研重仓股            机构批量减持
北向资金(如重仓港股)      5%    持续净流入                    持续净流出
```

### 3.3 基金吧情绪量化（独家）

国内基金吧（天天基金评论区）是散户情绪的富矿：

```python
def analyze_fund_bar_sentiment(fund_code):
    """抓取基金吧帖子，用LLM分析散户情绪"""
    posts = scrape_fund_forum(fund_code, days=7, limit=50)
    
    prompt = f"""以下是一只基金的基民讨论，请分析情绪：
    
    {posts}
    
    输出:
    - sentiment_score: -1到+1
    - key_topics: 大家在讨论什么(3个关键词)
    - panic_level: 0-10 恐慌程度
    - fomo_level: 0-10 追涨意愿
    """
```

---

## 四、辩论 Agent (DebateAgent) — 基金版 Prompt

### 4.1 Bull Prompt（看多方）

```
你是一位看好这只基金的买方分析师。请基于以下数据，尽全力为「买入/持有」找理由。

分析数据:
{fund_data_summary}

辩论历史（对方已说的话，你需要反驳）:
{previous_bear_argument}

请从以下角度组织你的论点（每个角度至少1条）：
1. 基金经理：为什么他/她值得信任
2. 持仓/穿透：底层资产质量如何
3. 市场时机：当前是否是好的入场点
4. 比较优势：为什么选这只而不是同类/指数

输出JSON: {{"arguments": ["论点1", "论点2", ...], "confidence": 0-1}}
```

### 4.2 Bear Prompt（看空方）

```
你是一位谨慎的基金分析师。请基于以下数据，尽全力为「不买/卖出」找理由。

分析数据:
{fund_data_summary}

辩论历史（对方已说的话）:
{previous_bull_argument}

请从以下角度组织你的论点：
1. 基金经理风险：离职风险、风格漂移、规模扩张过快
2. 持仓风险：集中度、行业暴露、估值偏贵
3. 费率/流动性：隐藏成本、赎回限制
4. 替代方案：同类型有更好的、或定投指数更优

输出JSON: {{"arguments": ["论点1", "论点2", ...], "confidence": 0-1}}
```

### 4.3 Judge Prompt（裁判）

```
你是一位中立的基金投资顾问。辩论结束了，请裁决。

Bull方论点: {bull_args}
Bear方论点: {bear_args}

做出最终裁决，考虑:
- 双向论点中，哪一方的论据更有数据支撑（而非情绪化）？
- 该基金适合什么类型的投资者（稳健/进取）？
- 如果BUY，建议一次性买入还是定投？
- 如果HOLD，等待什么信号出现后重新评估？
- 建议持有期：短期(<6月) / 中期(6-24月) / 长期(>2年)

输出JSON:
{{
    "final_signal": "BUY/SELL/HOLD",
    "confidence": 0-1,
    "reasoning": "200字裁决理由",
    "recommended_action": "如'定投每周¥2000，连续12周'",
    "target_position_pct": 0.0-1.0,
    "suggested_holding_period": "short/mid/long",
    "a_c_recommendation": "A类/C类",
    "watch_signals": ["需要观察的后续信号"]
}}
```

---

## 五、风控 Agent (RiskAgent) — 基金版规则

### 5.1 硬规则（按基金类型动态设定）

```python
FUND_RISK_RULES = {
    "偏股混合": {
        "max_position": 0.25,      # 单基金 ≤25%
        "max_drawdown": 0.12,      # 最大回撤 ≤12%
        "stop_loss": 0.10,         # 止损线 10%
        "min_holding_days": 90,    # 最短持有90天(赎回费惩罚)
        "申赎费检查": True,
    },
    "股票型": {
        "max_position": 0.20,
        "max_drawdown": 0.15,
        "stop_loss": 0.12,
        "min_holding_days": 90,
    },
    "指数型": {
        "max_position": 0.30,      # 指数更分散，可放宽
        "max_drawdown": 0.15,
        "stop_loss": 0.12,
        "min_holding_days": 30,    # ETF可T+1，更灵活
    },
    "偏债混合": {
        "max_position": 0.40,
        "max_drawdown": 0.06,
        "stop_loss": 0.04,
        "min_holding_days": 60,
    },
    "纯债": {
        "max_position": 0.50,
        "max_drawdown": 0.03,
        "stop_loss": 0.02,
        "min_holding_days": 30,
    },
    "QDII": {
        "max_position": 0.15,      # 海外风险+汇率风险，更保守
        "max_drawdown": 0.15,
        "stop_loss": 0.12,
        "min_holding_days": 90,    # QDII赎回T+7以上
    },
}
```

### 5.2 基金专属风险检查

```
额外硬规则（股票版没有的）：
1. 清盘风险: 规模 < 5000万 → 严重警告
2. 流动性风险: 近1月日均成交 < 100万 (ETF) 或 暂停赎回 → 否决
3. 费率陷阱: 持有<7天赎回费1.5% → 警告+建议延后
4. 经理变更风险: 基金经理任职<1年 → 降低仓位至5%
5. 风格漂移: 实际持仓与宣称类型偏离>30% → 严重警告
6. 集中度风险: 前3只重仓占总仓位>40% → 降仓位
```

---

## 六、执行 Agent (ExecutionAgent) — 基金版申赎单

### 6.1 申赎 vs 股票下单

```
股票下单:                   基金申赎:
  实时价格                     收盘净值(15:00前按当日)
  限价单/市价单                申购单/赎回单(净值成交)
  滑点成本                    申赎费率
  T+1/T+0                     普通T+1~T+2，QDII T+7~10
  最小1股                     最小1元起
```

### 6.2 执行逻辑

```python
def execute_fund_order(fund_code, signal, position_pct, portfolio_value, holding_period):
    if signal != "BUY":
        return {"status": "NO_ACTION"}
    
    amount = portfolio_value * position_pct
    
    # 15:00 时间窗判断
    from datetime import datetime
    now = datetime.now()
    nav_date = "today" if now.hour < 15 else "next_trading_day"
    
    # A/C 类选择
    if holding_period == "long":  # >1年
        share_class = "A"  # 申购费一次性，长期划算
    else:
        share_class = "C"  # 免申购费，短期划算
    
    # 费率计算
    if share_class == "A":
        subscribe_fee = amount * 0.0015  # 0.15% (1.5%打1折)
    else:
        subscribe_fee = 0
    
    # 定投模式
    if position_pct < 0.05:  # 仓小→定投
        order_type = "定投"
        dca_plan = f"每周定投 ¥{amount/12:.0f}，持续12周"
    else:
        order_type = "一次性申购"
        dca_plan = None
    
    return {
        "fund_code": fund_code,
        "amount": amount,
        "share_class": share_class,
        "nav_date": nav_date,
        "subscribe_fee": subscribe_fee,
        "order_type": order_type,
        "dca_plan": dca_plan,
        "warning": "持有<7天赎回费1.5%，请至少持有30天" if holding_period == "short" else None,
    }
```

---

## 七、回测方法论 — 基金版

### 7.1 数据源

```python
# akshare 可用基金数据
import akshare as ak

# 基金净值历史（日频）
nav = ak.fund_open_fund_info_em(fund="005827", indicator="单位净值走势")

# 基金持仓（季报，每季度更新）
holdings = ak.fund_portfolio_hold_detail_em(date="2025Q1")

# 同类基金排名
rank = ak.fund_open_fund_rank_em(symbol="混合型-偏股")
```

### 7.2 回测信号规则（简化版，省LLM成本）

```python
def generate_fund_signal(fund_code, date, nav_history, benchmark_history):
    """
    简化版信号（回测用，避免每条数据都调LLM）
    
    BUY条件:
      1. 净值 < SMA60 (相对低位)
      2. 近1月基金吧情绪从中性转正面
      3. 基金近1年同类排名前40%
      4. 满足≥2个条件 → BUY
    
    SELL条件:
      1. 净值偏离SMA60 > +20% (高位)
      2. 基金经理变更
      3. 连续2个季度跑输基准 >3%
      4. 满足≥1个条件 → SELL
    """
```

### 7.3 绩效指标

```
股票版指标(复用):          基金版新增:
  年化收益率 ✅              相对基准超额收益 ⭐
  夏普比率 ✅                信息比率(IR) ⭐
  最大回撤 ✅                卡玛比率(Calmar) ⭐
  索提诺比率 ✅              月度胜率 vs 基准 ✅
  胜率 ✅                    滚动1年收益分布 ✅
```

---

## 附录：完整数据源对照表

| 数据需求 | 美股(股票版) | 国内基金(基金版) | 备注 |
|---------|-------------|-----------------|------|
| 标的价格 | yfinance K线 | akshare 净值历史 | — |
| 基本面 | yfinance info | akshare + 天天基金 | 基金重仓股还需股票数据 |
| 基金经理 | — | akshare fund_manager | — |
| 持仓穿透 | — | akshare + yfinance/A股源 | 需获取底层股票数据 |
| 新闻情绪 | yfinance news + TextBlob | akshare新闻 + 天天基金吧 | 中文用LLM判 |
| 评级 | 分析师评级 | 晨星 + 天天基金评分 | — |
| 申赎资金 | — | 东方财富资金流向 | akshare已封装 |
| 实时行情 | yfinance streaming | —(基金无实时) | ETF可用akshare行情 |
| 比基准 | SPY | 沪深300/中证500/基金基准 | — |
| 费率 | — | akshare fund_info | — |
