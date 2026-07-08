# BioDynamics Agent v4 - 数值稳定性重试器（IB-007 修复）
# 对应 Issue Backlog IB-007：求解器发散/NaN 无参数级恢复。
#
# 问题背景：
#   当 ODE 求解器发散或产生 NaN 时，v3 仅重新生成相同代码重试，
#   无法从根本上解决数值不稳定问题。需要一个参数级恢复机制，
#   通过调整求解器参数（步长 / 方法 / 模型降阶）逐步尝试恢复。
#
# 设计原则（铁律）：
# 1. 纯模板替换，不调用 LLM（参数级恢复，无需重新生成代码）
# 2. 阶梯策略：从轻量（收紧步长）到重量（QSSA 降阶），逐步升级
# 3. 失败后标记 numerical_unstable_after_retry，供上游感知并降级处理
# 4. 不修改 sandbox.py（不可碰清单）
# 5. 策略可叠加：每次重试在上一轮修改基础上继续调整
#
# 阶梯策略（STRATEGIES）：
#   Attempt 1: 收紧 max_step（0.5→0.1）— 限制步长避免跳过快变区域
#   Attempt 2: 切换 BDF solver — 适用于刚性系统（隐式多步法）
#   Attempt 3: 切换 Radau solver — 高精度刚性求解器（隐式 Runge-Kutta）
#   Attempt 4: QSSA 降阶 — 准稳态近似，将快变量设为代数方程（dy=0）

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class NumericalStabilityRetry:
    """数值稳定性重试器（IB-007）。

    当 ODE 求解器发散或产生 NaN 时，按阶梯策略逐步调整求解参数，
    无需调用 LLM 重新生成代码。

    阶梯策略（共 4 级，由轻到重）：
      1. tighten_max_step: 收紧 max_step（0.5→0.1）
      2. switch_to_bdf: 切换 BDF solver（刚性系统）
      3. switch_to_radau: 切换 Radau solver（高精度刚性）
      4. qssa_reduction: QSSA 降阶（准稳态近似，快变量 dy=0）

    使用示例：
        retryer = NumericalStabilityRetry()
        result = retryer.retry(ode_code, "NaN detected", attempt=1)
        if result["exhausted"]:
            # 所有策略已用尽，标记失败
            ...
        else:
            # 使用 result["modified_code"] 重新执行
            ...
    """

    # 最大尝试次数（与 STRATEGIES 长度一致）
    MAX_ATTEMPTS = 4

    # 阶梯策略列表（按由轻到重排序）
    STRATEGIES = [
        {
            "name": "tighten_max_step",
            "description": "收紧 max_step（0.5→0.1）",
        },
        {
            "name": "switch_to_bdf",
            "description": "切换 BDF solver（刚性系统）",
        },
        {
            "name": "switch_to_radau",
            "description": "切换 Radau solver（高精度刚性）",
        },
        {
            "name": "qssa_reduction",
            "description": "QSSA 降阶（准稳态近似，快变量 dy=0）",
        },
    ]

    # 失败标记（写入代码以供上游感知）
    FAILURE_MARKER = "numerical_unstable_after_retry"

    def retry(self, ode_code: str, failure_reason: str, attempt: int) -> dict:
        """按阶梯策略对 ODE 代码应用数值稳定性恢复。

        Args:
            ode_code: 失败的 ODE Python 代码
            failure_reason: 失败原因（如 "NaN detected", "divergence"）
            attempt: 当前尝试编号（1-indexed，1=第一次重试策略）

        Returns:
            dict: {
                "modified_code": str,   # 修改后的代码（exhausted 时标记失败）
                "strategy": str,        # 应用的策略名（exhausted 时为 "exhausted"）
                "next_attempt": int,    # 下一次尝试编号
                "exhausted": bool,      # 是否已用尽所有策略
            }
        """
        logger.info(
            "数值稳定性重试：attempt=%d, reason=%s, MAX_ATTEMPTS=%d",
            attempt, failure_reason, self.MAX_ATTEMPTS,
        )

        # 策略已用尽：标记失败并返回原代码
        if attempt > self.MAX_ATTEMPTS:
            logger.warning(
                "数值稳定性重试已用尽所有策略（%d/%d），标记 %s",
                attempt - 1, self.MAX_ATTEMPTS, self.FAILURE_MARKER,
            )
            return {
                "modified_code": self._mark_failure(ode_code),
                "strategy": "exhausted",
                "next_attempt": attempt,
                "exhausted": True,
            }

        # 选择策略（attempt 为 1-indexed，STRATEGIES 为 0-indexed）
        strategy_idx = attempt - 1
        strategy = self.STRATEGIES[strategy_idx]
        logger.info(
            "应用策略 %d/%d：%s（%s）",
            attempt, self.MAX_ATTEMPTS, strategy["name"], strategy["description"],
        )

        # 应用模板替换
        modified_code = self._apply_strategy(ode_code, strategy["name"])

        return {
            "modified_code": modified_code,
            "strategy": strategy["name"],
            "next_attempt": attempt + 1,
            "exhausted": False,
        }

    # =========================================================================
    # 策略分发
    # =========================================================================
    def _apply_strategy(self, code: str, strategy_name: str) -> str:
        """根据策略名应用对应的模板替换。"""
        if strategy_name == "tighten_max_step":
            return self._tighten_max_step(code)
        elif strategy_name == "switch_to_bdf":
            return self._switch_solver(code, "BDF")
        elif strategy_name == "switch_to_radau":
            return self._switch_solver(code, "Radau")
        elif strategy_name == "qssa_reduction":
            return self._apply_qssa(code)
        else:
            logger.warning("未知策略：%s，返回原代码", strategy_name)
            return code

    # =========================================================================
    # 策略 1：收紧 max_step（0.5→0.1）
    # =========================================================================
    def _tighten_max_step(self, code: str) -> str:
        """策略 1：收紧 max_step（0.5→0.1）。

        匹配模式（覆盖模板中常见的写法）：
          - max_step=0.5 / max_step=1.0 / max_step=0.8 等 → max_step=0.1
          - max_step=MAX_STEP_MEDIUM / max_step=MAX_STEP_SLOW → max_step=0.1
          - MAX_STEP_MEDIUM = 0.5 → MAX_STEP_MEDIUM = 0.1
          - MAX_STEP_SLOW = 1.0 → MAX_STEP_SLOW = 0.1

        已收紧的 max_step=0.1 / max_step=0.01 保持不变。
        """
        modified = code

        # 1. max_step=<数值>：将 >= 0.5 的步长收紧为 0.1
        def _replace_numeric_step(match: re.Match) -> str:
            value_str = match.group(1)
            try:
                value = float(value_str)
                if value >= 0.5:
                    return "max_step=0.1"
            except ValueError:
                pass
            return match.group(0)

        modified = re.sub(
            r"max_step\s*=\s*(\d+\.?\d*)",
            _replace_numeric_step,
            modified,
        )

        # 2. max_step=MAX_STEP_MEDIUM / max_step=MAX_STEP_SLOW → max_step=0.1
        modified = re.sub(
            r"max_step\s*=\s*MAX_STEP_(?:MEDIUM|SLOW)\b",
            "max_step=0.1",
            modified,
        )

        # 3. MAX_STEP_MEDIUM = 0.5 → MAX_STEP_MEDIUM = 0.1
        modified = re.sub(
            r"(MAX_STEP_MEDIUM)\s*=\s*\d+\.?\d*",
            r"\1 = 0.1",
            modified,
        )

        # 4. MAX_STEP_SLOW = 1.0 → MAX_STEP_SLOW = 0.1
        modified = re.sub(
            r"(MAX_STEP_SLOW)\s*=\s*\d+\.?\d*",
            r"\1 = 0.1",
            modified,
        )

        # 添加策略标记注释
        modified = self._add_strategy_marker(
            modified, "tighten_max_step: max_step→0.1"
        )

        return modified

    # =========================================================================
    # 策略 2/3：切换求解器（BDF / Radau）
    # =========================================================================
    def _switch_solver(self, code: str, solver: str) -> str:
        """策略 2/3：切换 solve_ivp 求解器（BDF / Radau）。

        匹配 solve_ivp 调用中的 method 参数，替换为指定求解器：
          - method="LSODA" → method="<solver>"
          - method='LSODA' → method='<solver>'
          - method="BDF" → method="<solver>"（策略 3 时 BDF→Radau）
          - method="Radau" → method="<solver>"（理论上不会出现，防御性匹配）

        BDF：隐式多步法，适用于刚性系统，效率较高
        Radau：隐式 Runge-Kutta，高精度刚性求解器，计算成本更高
        """
        # 替换所有 method="..." 为指定 solver（兼容单双引号）
        modified = re.sub(
            r'''method\s*=\s*["'](?:LSODA|BDF|Radau|RK45|RK23|DOP853)["']''',
            f'method="{solver}"',
            code,
        )

        # 添加策略标记注释
        modified = self._add_strategy_marker(
            modified, f"switch_solver: method→{solver}"
        )

        return modified

    # =========================================================================
    # 策略 4：QSSA 降阶（准稳态近似）
    # =========================================================================
    def _apply_qssa(self, code: str) -> str:
        """策略 4：QSSA 降阶（准稳态近似）。

        将快变量的导数设为 0，使其成为代数变量：
          - 准稳态假设：快反应（磷酸化 / 激活 / 剪切）瞬间达到平衡
          - 数学等价：dy_fast/dt = 0 → 快变量由代数方程约束
          - 效果：降低系统刚性，避免数值发散

        快变量识别（启发式）：
          - 以 "p" 开头且第二字符大写（磷酸化形式，如 pRb / pEGFR）
          - 以 "_active" 结尾（激活形式，如 Caspase3_active）
          - 以 "Cleaved" 开头（剪切形式，如 CleavedPARP）

        实现：在 _ode_rhs 的 return dy 前插入快变量清零代码块。
        """
        # QSSA 快变量清零代码（插入到 _ode_rhs 的 return dy 之前）
        qssa_block = (
            "    # ===== QSSA 降阶（IB-007 策略 4）=====\n"
            "    # 准稳态近似：快变量（磷酸化 / active / Cleaved）设为代数方程 dy=0\n"
            "    for _qssa_sp, _qssa_idx in SP_IDX.items():\n"
            "        if _qssa_sp.startswith('p') and len(_qssa_sp) > 1 and _qssa_sp[1:2].isupper():\n"
            "            dy[_qssa_idx] = 0.0\n"
            "        elif _qssa_sp.endswith('_active'):\n"
            "            dy[_qssa_idx] = 0.0\n"
            "        elif _qssa_sp.startswith('Cleaved'):\n"
            "            dy[_qssa_idx] = 0.0\n"
        )

        modified = code

        # 在 _ode_rhs 函数体的 return dy 前插入 QSSA 代码块
        # 匹配 def _ode_rhs 到其首个 return dy（非贪婪，DOTALL 模式）
        pattern_rhs = r"(def\s+_ode_rhs\s*\([^)]*\)[^:]*:.*?)(return\s+dy\b)"
        match_rhs = re.search(pattern_rhs, modified, re.DOTALL)

        if match_rhs:
            # 在 return dy 前插入 QSSA 块
            insert_pos = match_rhs.start(2)
            modified = modified[:insert_pos] + qssa_block + modified[insert_pos:]
            logger.info("QSSA 降阶：已在 _ode_rhs 中插入快变量清零代码")
        else:
            # 未找到 _ode_rhs，尝试通用的首个 return dy
            pattern_fallback = r"(return\s+dy\b)"
            match_fallback = re.search(pattern_fallback, modified)
            if match_fallback:
                insert_pos = match_fallback.start()
                modified = modified[:insert_pos] + qssa_block + modified[insert_pos:]
                logger.info("QSSA 降阶：已在首个 return dy 前插入快变量清零代码（fallback）")
            else:
                logger.warning("QSSA 降阶：未找到 return dy，仅添加标记")

        # 添加策略标记注释
        modified = self._add_strategy_marker(
            modified, "qssa_reduction: 快变量 dy=0（准稳态近似）"
        )

        return modified

    # =========================================================================
    # 辅助方法
    # =========================================================================
    def _add_strategy_marker(self, code: str, marker: str) -> str:
        """在代码顶部添加策略标记注释（便于追踪应用的策略）。"""
        marker_line = f"# [NumericalStabilityRetry] {marker}\n"
        return marker_line + code

    def _mark_failure(self, code: str) -> str:
        """标记代码为数值不稳定（所有重试策略已用尽）。

        添加两层标记确保上游可感知：
          1. 顶部注释标记（便于代码审查）
          2. print 语句（写入 stdout，供 sandbox _classify_error 感知）
        """
        failure_comment = f"# [NumericalStabilityRetry] {self.FAILURE_MARKER}\n"
        failure_print = f'print("{self.FAILURE_MARKER}")\n'

        # 在代码顶部添加注释标记
        marked = failure_comment + code

        # 在代码中（main 块前）添加 print 语句，确保执行时输出标记
        main_pattern = r'(if\s+__name__\s*==\s*["\']__main__["\']\s*:)'
        match = re.search(main_pattern, marked)
        if match:
            insert_pos = match.start()
            marked = marked[:insert_pos] + failure_print + "\n" + marked[insert_pos:]
        else:
            # 无 main 块，追加到末尾
            marked = marked + "\n" + failure_print

        return marked


__all__ = ["NumericalStabilityRetry"]
