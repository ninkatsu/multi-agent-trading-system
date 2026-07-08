#!/bin/bash
# ============================================
# 基金分析服务启动脚本
# 支持国内公募基金 (005827 易方达蓝筹, 110011 易方达中小盘...)
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
if [ -d "fund/.venv" ]; then
    echo "🔧 使用现有虚拟环境 fund/.venv"
    source fund/.venv/bin/activate 2>/dev/null || source fund/.venv/Scripts/activate 2>/dev/null || true
    PYTHON="fund/.venv/bin/python"
    if [ ! -f "$PYTHON" ]; then
        PYTHON="fund/.venv/Scripts/python.exe"
    fi
else
    echo "🔧 创建虚拟环境..."
    python3 -m venv fund/.venv
    source fund/.venv/bin/activate 2>/dev/null || source fund/.venv/Scripts/activate 2>/dev/null || true
    PYTHON="fund/.venv/bin/python"
    if [ ! -f "$PYTHON" ]; then
        PYTHON="fund/.venv/Scripts/python.exe"
    fi
    echo "📦 安装共享依赖..."
    $PYTHON -m pip install -q -r shared/requirements.txt
    echo "📦 安装基金版依赖..."
    $PYTHON -m pip install -q -r fund/requirements.txt
fi

# ---------- 启动 ----------
echo ""
echo "============================================"
echo "  基金多Agent投资决策系统"
echo "============================================"
echo ""
echo "  数据源: akshare (国内免费金融数据库)"
echo "  支持: 国内公募基金 6位代码"
echo "  示例: 005827 (易方达蓝筹精选)"
echo "        110011 (易方达中小盘)"
echo "        163406 (兴全合润)"
echo "        161725 (招商中证白酒)"
echo "============================================"
echo ""

cd fund
$PYTHON -m graph.fund_graph
