@echo off
chcp 65001 >nul
title BioDynamics Agent - 大模型连通性探测

echo ========================================
echo  BioDynamics Agent - 大模型连通性探测
echo ========================================
echo.

REM 切换到脚本所在目录的上级目录（backend 根目录）
cd /d "%~dp0"

REM 使用当前环境已有的 Python 解释器；若未找到则尝试 py launcher
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo 未检测到 python，尝试 py launcher ...
    py scripts\test_model_connectivity.py
) else (
    python scripts\test_model_connectivity.py
)

echo.
echo 按任意键退出 ...
pause >nul
