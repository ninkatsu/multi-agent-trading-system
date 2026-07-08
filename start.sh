#!/bin/bash
# ============================================
# 多Agent量化交易系统 - 统一启动脚本
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  多Agent量化交易与投资决策系统"
echo "============================================"
echo ""
echo "  1) 股票分析 (美股/A股)"
echo "  2) 基金分析 (国内公募基金)"
echo "  3) 退出"
echo ""
read -p "请选择 [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "启动股票分析服务..."
        bash start_stock.sh
        ;;
    2)
        echo ""
        echo "启动基金分析服务..."
        bash start_fund.sh
        ;;
    3)
        echo "再见!"
        exit 0
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
