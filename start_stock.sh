#!/bin/bash
# ============================================
# 股票分析服务启动脚本
# 支持美股 (AAPL, MSFT, GOOGL...) 和 A股 (600519.SS...)
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 环境检查 ----------
if [ ! -f ".env" ]; then
    echo "⚠️  未找到 .env 文件，从 .env.example 复制..."
    cp .env.example .env
    echo "✅ 已创建 .env，请编辑填入你的 DeepSeek API Key 后重新运行"
    echo "   编辑命令: nano .env  或  vim .env"
    exit 1
fi

# 检查 Python 虚拟环境
if [ -d "stock/.venv" ]; then
    echo "🔧 使用现有虚拟环境 stock/.venv"
    source stock/.venv/bin/activate 2>/dev/null || source stock/.venv/Scripts/activate 2>/dev/null || true
    PYTHON="stock/.venv/bin/python"
    if [ ! -f "$PYTHON" ]; then
        PYTHON="stock/.venv/Scripts/python.exe"  # Windows Git Bash
    fi
else
    echo "🔧 创建虚拟环境..."
    python3 -m venv stock/.venv
    source stock/.venv/bin/activate 2>/dev/null || source stock/.venv/Scripts/activate 2>/dev/null || true
    PYTHON="stock/.venv/bin/python"
    if [ ! -f "$PYTHON" ]; then
        PYTHON="stock/.venv/Scripts/python.exe"
    fi
    echo "📦 安装依赖..."
    $PYTHON -m pip install -q -r shared/requirements.txt
fi

# ---------- 启动 ----------
echo ""
echo "============================================"
echo "  股票多Agent投资决策系统"
echo "============================================"
echo ""
echo "  数据源: Yahoo Finance (yfinance)"
echo "  支持: 美股 ticker (AAPL, MSFT, TSLA...)"
echo "        A股 (需加后缀: 600519.SS, 000858.SZ)"
echo "============================================"
echo ""

cd stock
$PYTHON -m graph.trading_graph
