# Multi-Agent Trading System | 多Agent量化交易与投资决策系统

> **6个AI Agent协作做投资决策** — 同时支持**股票**（美股/A股）和**基金**（国内公募基金），用 LLM 辩论机制做出审慎的投资判断。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek_V4_Pro-536DFE.svg)](https://platform.deepseek.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 这个项目是什么？

一个**企业级多Agent量化交易系统**，模拟真实投资公司的决策流程：

```
你给它一个标的（如 AAPL 或 005827）
  → 3个分析师Agent同时工作
  → 牛方和熊方展开辩论（强制从多空角度思考）
  → 风控官审批（拥有一票否决权）
  → 通过才执行下单（模拟交易，不用真钱）
```

**两大服务平级运行：**

| 服务 | 标的 | 数据源 | 定位 | 频率 |
|------|------|--------|------|------|
| `stock/` | 美股/A股 | yfinance | **高频分析工具** | 每天/每周 |
| `fund/` | 国内公募基金 | akshare（免费） | **季度决策+日常监控+定投辅助** | 三模式 |

> ⚠️ **重要**: 基金持仓每季度公布一次，季报空窗期（第2-3个月）持仓可能已大幅变化。因此基金版不支持日常交易信号，改为三模式系统。

---

## 系统架构

```
                    ┌──────────────────────────┐
                    │    shared/ 共用配置层     │
                    │  LLM工厂 / 设置 / JSON   │
                    └──────────┬───────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
    ┌─────────▼─────────┐           ┌──────────▼──────────┐
    │   stock/ 股票版    │           │   fund/ 基金版       │
    │                   │           │                      │
    │ 基本面 → PE/PB/ROE│           │ 基金质量 → 经理/穿透 │
    │ 技术面 → K线/MACD │           │ 技术面 → 净值趋势    │
    │ 情绪面 → 英文新闻  │           │ 情绪面 → 中文新闻(GLM)│
    │ 辩论 → Bull/Bear  │           │ 辩论 → 复用框架      │
    │ 风控 → 硬规则+LLM │           │ 风控 → 类型动态阈值   │
    │ 执行 → 限价单/滑点│           │ 执行 → 申赎单/定投   │
    └───────────────────┘           └──────────────────────┘
```

**三大创新点：**
1. **Bull/Bear 辩论机制** — 强制从多空两个角度审视，避免确认偏误
2. **风控双层门控** — 硬规则（确定性代码，不可绕过）+ LLM软判断
3. **基金版穿透分析** — 不只看基金净值，还穿透到底层股票的加权PE/ROE

---

## 快速开始（3步跑起来）

### 环境要求
- Python 3.11+
- DeepSeek API Key（[免费注册](https://platform.deepseek.com)）

### Step 1: 配置 API Key

```bash
git clone git@github.com:ninkatsu/multi-agent-trading-system.git
cd multi-agent-trading-system

# 复制环境变量模板
cp .env.example .env

# 编辑 .env，填入你的 DeepSeek API Key
# DEEPSEEK_API_KEY=sk-your-key-here
```

### Step 2: 安装依赖

```bash
# 股票版
cd stock
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r ../shared/requirements.txt
cd ..

# 基金版（需要两个都跑的话）
cd fund
python -m venv .venv
source .venv/bin/activate
pip install -r ../shared/requirements.txt -r requirements.txt
cd ..
```

### Step 3: 运行

```bash
# 启动股票分析
bash start_stock.sh
# 输入股票代码，如 AAPL / MSFT / 600519.SS

# 启动基金分析（推荐直接使用三模式命令）
python run_fund_analysis.py quarterly 005844    # 季度深度分析（一年4次）
python run_fund_analysis.py monitor 005844      # 日常监控告警（每周）
python run_fund_analysis.py dca 005844 3000     # 定投时点评估（每周）

# 或统一菜单
bash start.sh
```

---

## 基金版三模式系统

> **为什么不和股票一样天天分析？** 基金持仓只在季报公布，空窗期长达3个月。期间持仓可能面目全非，基于过期数据的分析反而不如不做。

| 模式 | 命令 | 频率 | LLM调用 | 作用 |
|------|------|------|---------|------|
| **季度深度** | `quarterly <代码>` | 每年4次 | 6个Agent全链路 | 持仓新鲜时做穿透分析+辩论+决策 |
| **日常监控** | `monitor <代码>` | 每周1次 | 无（纯数据检查） | 8项硬信号告警：经理变更/清盘/AUM异常/净值异动 |
| **定投助手** | `dca <代码> [金额]` | 每周1次 | 无（纯公式评分） | 5档定投信号：加倍/正常/减半/暂停/观望 |

**示例输出：**

```bash
$ python run_fund_analysis.py monitor 005844

============================================================
  基金监控报告: 东方人工智能主题混合A (005844)
============================================================
  🟡 1个黄色预警

  🟢 基金经理 — 严凯任职6.2年，稳定
  🟢 基金规模 — 119.9亿，合理区间
  🟡 净值异常 — 近20天5次>5%暴涨（可能巨额赎回导致）
  🟢 清盘风险 — 无
  🟢 费率 — 1.5%，正常
```

```bash
$ python run_fund_analysis.py dca 005844 3000

============================================================
  定投评估: 东方人工智能主题混合A (005844)
============================================================
  信号: ⏸️ 暂停定投
  评分: 4.0/10
  建议: ¥0/周（暂停，资金保留等待更好时机）

  净值位置 (0/3): 偏离SMA60=43%，历史高位
  回撤情况 (1/3): 年内回撤-9.8%
  市场情绪 (1/2): 恐贪指数50（中性）
  波动率   (2/2): 60.7%（高波动，定投优势明显但需等低位）
```

---

## 六个Agent详解

### 股票版 (stock/)

#### 1. FundamentalAgent — 基本面分析
分析 PE/PB/ROE/营收增长/利润率/自由现金流
```
输入: AAPL → 拉取 yfinance 财报 → LLM 评估 → 输出 1-10 评分
```

#### 2. TechnicalAgent — 技术面分析
K线形态、MACD/RSI/布林带/SMA/成交量
```
输入: AAPL → 6个月K线 → pandas_ta 计算指标 → LLM 综合研判
```

#### 3. SentimentAgent — 情绪面分析
新闻NLP(TextBlob)、机构持仓、分析师评级
```
输入: AAPL → 抓取新闻标题 → TextBlob 极性分析 → LLM 评分
```

#### 4. DebateAgent — 牛熊辩论 ⭐
Bull方找买入理由 → Bear方找卖出理由 → 2轮交锋 → Judge裁决
```
这是最大创新点 — 强制对抗性辩论，避免确认偏误
```

#### 5. RiskAgent — 风控守门 ⭐
硬规则(仓位≤10%, VaR≤2%, 回撤≤8%) + LLM软判断 → 一票否决权
```
硬规则用确定性代码，LLM处理边界情况 — 金融系统的安全底线原则
```

#### 6. ExecutionAgent — 执行下单
限价单控制滑点、模拟成交(Dry Run) / Alpaca Paper Trading
```
风控批准后才执行 | Dry Run模式纯日志，无真实资金风险
```

### 基金版 (fund/)

基金版和股票版拓扑一致，但每个Agent的"金融内核"换成了基金语境：

| Agent | 股票版 | 基金版 |
|-------|--------|--------|
| 基本面 | PE/PB/ROE | **基金经理**（任期/年化/回撤）+ **穿透分析**（加权PE/ROE） |
| 技术面 | K线+MACD/RSI | **净值趋势**（日频）+ 定投择时信号 |
| 情绪面 | TextBlob英文NLP | **LLM直接判中文情绪** + 申赎资金流向 |
| 辩论 | Bull/Bear/Judge | 复用框架，prompt换基金语境 |
| 风控 | 固定阈值(10%) | **按基金类型动态阈值** + 清盘/费率/T+1检查 |
| 执行 | 限价单+滑点 | **申购/赎回单** + A/C类选择 + 定投计划 |

#### 基金版独有的亮点：

1. **穿透分析**（你的想法被实现了！）：不只看基金，还看底层持仓——
   ```
   加权PE = Σ(每只重仓股的PE × 持仓权重)
   加权ROE = Σ(每只重仓股的ROE × 持仓权重)
   ```
   这样买基金前就知道"你的钱到底投进了什么质量的资产"

2. **4423法则初筛**：先用经典框架快速过滤，再过详细评分

3. **定投智能建议**：不只输出"买/不买"，还输出"定投每周¥2000，连续12周"等具体方案

4. **A/C类自动选择**：根据建议持有期自动判断买A类还是C类更省钱

5. **15:00时间窗判断**：自动判断按哪天净值成交

---

## 配置说明 (.env)

```bash
# LLM: DeepSeek (默认)
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-key        # 在 platform.deepseek.com 获取
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 备选: 也可用 Anthropic 兼容端点
# DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic
# LLM_PROVIDER=anthropic

# 风控参数 (股票版)
MAX_POSITION_SIZE=0.1               # 单票≤10%
MAX_DRAWDOWN_LIMIT=0.08             # 回撤≤8%

# 风控参数 (基金版)
FUND_MAX_POSITION_EQUITY=0.25       # 偏股基金≤25%
FUND_MAX_DRAWDOWN_EQUITY=0.12       # 偏股回撤≤12%
```

**两个 DeepSeek 端点都可用**（已测试）：
- `https://api.deepseek.com/v1` — OpenAI 兼容（推荐，更稳定）
- `https://api.deepseek.com/anthropic` — Anthropic 兼容（备选）

---

## 项目结构

```
multi-agent-trading-system/
│
├── README.md                    ← 你正在看的文件
├── .env.example                 ← 环境变量模板
├── start.sh                     ← 统一启动菜单
├── start_stock.sh               ← 股票版启动
├── start_fund.sh                ← 基金版启动
│
├── shared/                      ← 🔧 共用配置层
│   ├── config/
│   │   ├── settings.py          ← 集中配置 (LLM/风控/回测)
│   │   ├── llm.py               ← LLM工厂 (DeepSeek/OpenAI/Anthropic)
│   │   └── json_utils.py        ← 容错JSON解析
│   └── requirements.txt         ← 共享依赖
│
├── stock/                       ← 📈 股票版
│   ├── agents/                  ← 6个Agent
│   ├── graph/trading_graph.py   ← LangGraph编排
│   ├── tools/                   ← yfinance封装 + 技术指标
│   ├── backtest/                ← 回测引擎
│   └── config/                  ← 兼容层(→shared/)
│
├── fund/                        ← 🏦 基金版
│   ├── agents/                  ← 6个基金Agent
│   │   ├── fund_quality_agent.py    ← 基金经理+穿透分析
│   │   ├── technical_agent.py       ← 净值趋势+定投信号
│   │   ├── sentiment_agent.py       ← 中文NLP(LLM)情绪
│   │   ├── debate_agent.py          ← 基金版辩论
│   │   ├── risk_agent.py            ← 类型动态阈值风控
│   │   └── execution_agent.py       ← 申赎单+定投计划
│   ├── graph/fund_graph.py      ← LangGraph编排(复用拓扑)
│   ├── tools/fund_data.py       ← akshare数据封装
│   ├── backtest/                ← 基金回测引擎
│   └── config/                  ← 兼容层(→shared/)
│
└── docs/                        ← 📚 文档
    ├── 金融背景/                  ← 金融概念从零讲解
    └── 基金决策/                  ← 基金改造方案 + 评分模型
        ├── 01-基金vs股票场景对比与改造方案.md
        ├── 02-基金专属背景知识.md
        └── 03-量化评分模型与Agent Prompt设计.md  ← 评分公式+Prompt模板
```

---

## 使用示例

### 股票分析

```
$ bash start_stock.sh

请输入股票代码 (默认 AAPL): MSFT

正在分析 MSFT...

📊 分析结果:
  [fundamental] 评分: 7.5/10 | 信号: BUY
    理由: PE 32偏高但ROE 45%优秀，云业务增长强劲，FCF充沛
  [technical] 评分: 6/10 | 信号: HOLD
    理由: RSI 58中性，MACD即将金叉，价格在SMA50上方
  [sentiment] 评分: 7/10 | 信号: BUY
    理由: 新闻情绪偏正面，机构持续增持，AI主题热度高

🗣️ 辩论结论: BUY (置信度: 72%)
  理由: Bull方强调AI业务增长和强劲现金流，Bear方担忧估值偏高...
  建议: 建仓5%，设止损$420

🛡️ 风控: 通过
  VaR(95%): 1.8%

💰 执行: FILLED_DRY_RUN
  [模拟] BUY 120股 MSFT @ $428.50
```

### 基金分析

```
$ bash start_fund.sh

请输入基金代码 (默认 005827 易方达蓝筹): 005827

正在分析基金 005827...

📊 分析结果:
  [fund_quality] 评分: 6.5/10 | 信号: HOLD
    基金经理: 张坤, 穿透PE: 22.5, 穿透ROE: 0.18
    4423初筛: ✅通过
  [technical] 评分: 5/10 | 信号: HOLD
    净值在SMA60附近横盘，定投评分: 6/10
  [sentiment] 评分: 4/10 | 信号: SELL
    新闻情绪偏负面 -0.3，资金持续净赎回

🗣️ 辩论结论: HOLD (置信度: 60%)
  裁决理由: 经理能力强但规模偏大，持仓集中...
  建议行动: 等净值回调至SMA60附近再考虑，或启动小额定投
  A/C类建议: A | 持有期: long

🛡️ 风控: 通过 (基金类型: 偏股混合)
  仓位建议从15%调整至10%

💰 执行: PENDING_DRY_RUN
  [模拟] 申购 易方达蓝筹(005827) A类 ¥100,000 | 申购费¥150
```

---

## 数据源说明

| 数据 | 股票版 | 基金版 | 费用 |
|------|--------|--------|------|
| 行情/K线 | yfinance | — | 免费 |
| 净值 | — | akshare | 免费 |
| 估值指标 | yfinance info | akshare + 穿透计算 | 免费 |
| 基金经理 | — | akshare / 天天基金 | 免费 |
| 持仓数据 | — | akshare（季报） | 免费 |
| 新闻 | yfinance news | akshare / 东方财富 | 免费 |
| 情绪NLP | TextBlob | LLM直接判断 | API调用 |
| 交易执行 | Alpaca Paper | 无公开API（手动） | Alpaca免费注册 |

> ⚠️ **基金版注意**：国内公募基金没有公开的交易API。系统会输出具体的申购建议（金额/A-C类/定投方案），但目前需要手动在天天基金/支付宝操作。

---

## 文档索引

### 金融背景（零基础入门）
- [01-基本面指标真正含义](docs/金融背景/01-基本面指标真正含义.md)
- [02-技术面指标真正含义](docs/金融背景/02-技术面指标真正含义.md)
- [03-情绪面与市场心理](docs/金融背景/03-情绪面与市场心理.md)
- [04-风控核心概念](docs/金融背景/04-风控核心概念.md)
- [05-回测与绩效评估](docs/金融背景/05-回测与绩效评估.md)

### 基金决策（改造方案）
- [01-基金vs股票场景对比与改造方案](docs/基金决策/01-基金vs股票场景对比与改造方案.md)
- [02-基金专属背景知识](docs/基金决策/02-基金专属背景知识.md)
- [03-量化评分模型与Agent Prompt设计](docs/基金决策/03-量化评分模型与Agent-Prompt设计.md) ⭐

---

## 参考项目与致谢

| 项目 | 特点 |
|------|------|
| [TradingAgents](https://github.com/TauricResearch/TradingAgents) | 多Agent金融交易框架 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agent编排框架 |
| [akshare](https://github.com/akfamily/akshare) | 国内开源金融数据库 |

---

## 免责声明

- 本项目仅用于**学习和研究**，不构成任何投资建议
- 回测结果不代表未来表现
- 请勿用于真实交易（除非你完全了解风险）
- 基金交易请通过正规渠道（天天基金/支付宝/券商）
