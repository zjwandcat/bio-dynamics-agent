# BioDynamics Agent v4 - Simulation Planner Agent（Phase 6 / Task 6.6.3）
#
# 选仿真类型 / 求解器 / 多时间尺度。
#
# 职责：
# 1. 根据通路类别与 ODE 系统特征选择仿真类型（ode/dde/stochastic）
# 2. 根据仿真类型选择求解器（scipy.solve_ivp/dde_solver/gillespie）
# 3. 多通路场景 → 多时间尺度（fast: 0-1min, slow: 0-120min）
# 4. 输出 v4_simulation_plan（不修改 v3 execution_result）
#
# 设计原则（铁律）：
# 1. Feature Flag V4_DYNAMIC_ROUTING_ENABLED=false → 返回 {}（不执行）
# 2. 不修改 v3 execution_result / ode_model；仅新增 v4_simulation_plan
# 3. 失败降级：任何异常返回默认 plan（simulation_type="ode"）
# 4. DDE 通路 → simulation_type="dde", solver="dde_solver"
# 5. 低分子数 → simulation_type="stochastic", solver="gillespie"
#
# 参考：
# - app.ode_renderer_v2._OSCILLATORY_PATHWAYS（振荡通路 → DDE）
# - app.solvers.dde_solver（DDE 求解器）
# - tasks.md SubTask 6.6.3

from __future__ import annotations

import logging
import re
from typing import Any

# app.config 无循环依赖风险（不导入 agents_v4），可在模块级导入
from app.config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# 常量
# =============================================================================
# 振荡通路类别集合（需 DDE 求解器），与 mechanism_builder.py 对齐
_OSCILLATORY_PATHWAYS: set[str] = {
    "p53", "p53_signaling",
    "NF_KB", "NF_kB", "nf_kb",
    "TGF_BETA", "TGF_beta", "tgf_beta",
    "JAK_STAT", "JAK-STAT",
}

# 合法的仿真类型
_SIMULATION_TYPES: set[str] = {"ode", "dde", "stochastic"}

# 合法的求解器
_SOLVERS: set[str] = {"scipy.solve_ivp", "dde_solver", "gillespie"}

# 仿真类型 → 求解器 映射
_SIM_TYPE_TO_SOLVER: dict[str, str] = {
    "ode": "scipy.solve_ivp",
    "dde": "dde_solver",
    "stochastic": "gillespie",
}

# 低分子数阈值（低于此值建议使用随机仿真）
_LOW_MOLECULE_THRESHOLD: float = 1000.0


class SimulationPlannerAgent:
    """v4 仿真规划 Agent：选仿真类型 / 求解器 / 多时间尺度。

    根据通路类别与 ODE 系统特征选择仿真类型与求解器，
    多通路场景输出多时间尺度计划。

    用法::

        agent = SimulationPlannerAgent()
        update = agent.plan(state)
        # update = {"v4_simulation_plan": {simulation_type, solver, ...}}
    """

    AGENT_VERSION: str = "v4.0"

    # -------------------------------------------------------------------------
    # 主入口
    # -------------------------------------------------------------------------
    def plan(self, state: dict) -> dict:
        """主入口：规划仿真类型、求解器与时间尺度。

        Args:
            state: LangGraph 全局状态，读取：
                - ``v4_ode_system``: ODE 系统输出（含 dde_delay_minutes）
                - ``v4_pathway_class``: 通路类别字符串
                - ``v4_reaction_ir``: Reaction IR（用于检测低分子数）

        Returns:
            flag=false 时返回 {}
            正常时返回 ``{"v4_simulation_plan": {...}}``
            失败时返回默认 plan ``{"v4_simulation_plan": {simulation_type: "ode", ...}}``

            v4_simulation_plan 结构::

                {
                    "simulation_type": str,    # "ode" / "dde" / "stochastic"
                    "solver": str,             # "scipy.solve_ivp" / "dde_solver" / "gillespie"
                    "t_start": float,          # 仿真起始时间（分钟）
                    "t_end": float,            # 仿真结束时间（分钟）
                    "n_points": int,           # 输出时间点数
                    "time_scales": list[dict],  # 多时间尺度（多通路时）
                    "multi_pathway": bool,     # 是否为多通路场景
                }
        """
        # 1. Feature Flag 检查（铁律：flag=false 不执行）
        if not settings.effective_v4_dynamic_routing_enabled():
            logger.debug("V4_DYNAMIC_ROUTING_ENABLED effective=false，SimulationPlanner 跳过")
            return {}

        try:
            # 2. 提取输入
            ode_system = state.get("v4_ode_system") or {}
            pathway_class = state.get("v4_pathway_class", "") or ""
            reaction_ir = state.get("v4_reaction_ir") or {}

            # 3. 判断仿真类型
            is_oscillatory = self._is_oscillatory(pathway_class)
            is_multi_pathway = self._is_multi_pathway(pathway_class)
            is_low_molecule = self._check_low_molecule(reaction_ir)

            declared_dde = float(ode_system.get("dde_delay_minutes", 0.0) or 0.0) > 0.0

            # 优先级：显式 DDE/振荡 > stochastic > ode。Wnt 等非振荡通路也可
            # 通过 PathwayGraph.temporal 声明转录反馈延迟。
            if declared_dde or is_oscillatory:
                simulation_type = "dde"
            elif is_low_molecule:
                simulation_type = "stochastic"
            else:
                simulation_type = "ode"

            # 4. 选择求解器
            solver = _SIM_TYPE_TO_SOLVER.get(simulation_type, "scipy.solve_ivp")

            # 5. 确定时间范围
            explicit_duration = self._extract_duration_minutes(state)
            t_start, t_end = self._determine_time_range(
                ode_system, is_multi_pathway, explicit_duration
            )
            n_points = 200

            # 6. 确定时间尺度
            time_scales = self._build_time_scales(is_multi_pathway, t_end)
            rtol, atol = self._solver_tolerances(state)

            # 7. 构建计划
            sim_plan: dict[str, Any] = {
                "simulation_type": simulation_type,
                "solver": solver,
                "t_start": t_start,
                "t_end": t_end,
                "n_points": n_points,
                "time_scales": time_scales,
                "multi_pathway": is_multi_pathway,
                "rtol": rtol,
                "atol": atol,
            }

            logger.info(
                "SimulationPlanner: type=%s solver=%s t_end=%.1f multi=%s",
                simulation_type, solver, t_end, is_multi_pathway,
            )

            return {"v4_simulation_plan": sim_plan}

        except Exception as exc:
            # 失败降级：返回默认 plan，不阻塞流水线
            logger.warning("SimulationPlanner.plan 失败，降级返回默认 plan: %s", exc)
            return {
                "v4_simulation_plan": self._default_plan(),
            }

    def generate(self, state: dict) -> dict:
        """DynamicRouter 调度入口（别名，委托给 plan）。"""
        return self.plan(state)

    # -------------------------------------------------------------------------
    # 内部辅助方法
    # -------------------------------------------------------------------------
    @staticmethod
    def _is_oscillatory(pathway_class: str) -> bool:
        """判断是否为振荡通路（p53/NF_KB/TGF_beta/JAK_STAT）→ DDE。"""
        if not pathway_class:
            return False
        pc_lower = pathway_class.lower()
        for osc in _OSCILLATORY_PATHWAYS:
            if osc.lower() in pc_lower:
                return True
        return False

    @staticmethod
    def _is_multi_pathway(pathway_class: str) -> bool:
        """判断是否为多通路场景。

        多通路的 pathway_class 格式为 "MULTI:pathway1+pathway2" 或包含 "+"。
        """
        if not pathway_class:
            return False
        return pathway_class.startswith("MULTI:") or "+" in pathway_class

    @staticmethod
    def _check_low_molecule(reaction_ir: dict) -> bool:
        """检查是否存在低分子数场景（建议使用随机仿真）。

        判断条件：任何 species 的 concentration_unit 为 "molecule_per_cell"
        且 initial_concentration < 阈值（1000 分子）。
        """
        species_list = reaction_ir.get("species", []) or []
        for sp in species_list:
            if not isinstance(sp, dict):
                continue
            unit = sp.get("concentration_unit", "nM")
            if unit == "molecule_per_cell":
                try:
                    ic = float(sp.get("initial_concentration", 0.0))
                    if 0 < ic < _LOW_MOLECULE_THRESHOLD:
                        return True
                except (TypeError, ValueError):
                    continue
        return False

    @staticmethod
    def _extract_duration_minutes(state: dict) -> float | None:
        """Read a governed benchmark/user duration without coupling to one schema."""
        candidates: list[Any] = [
            state.get("benchmark_duration"),
            state.get("simulation_duration"),
        ]
        for container_name in ("benchmark_case", "benchmark_spec", "benchmark_input"):
            container = state.get(container_name)
            if not isinstance(container, dict):
                continue
            candidates.append(container.get("duration"))
            input_spec = container.get("input")
            if isinstance(input_spec, dict):
                candidates.append(input_spec.get("duration"))
        user_input = str(state.get("user_input") or "")
        match = re.search(r"\bduration\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*min\b", user_input, re.I)
        if match:
            candidates.insert(0, match.group(1))
        for candidate in candidates:
            try:
                duration = float(candidate)
            except (TypeError, ValueError):
                continue
            if duration > 0:
                return duration
        return None

    @staticmethod
    def _solver_tolerances(state: dict) -> tuple[float, float]:
        """Use high precision for L5/C4-guided runs and preserve legacy defaults otherwise."""
        difficulty = str(state.get("benchmark_difficulty") or state.get("difficulty") or "")
        hints: list[str] = [str(state.get("optimization_hints") or "")]
        for container_name in ("benchmark_case", "benchmark_spec"):
            container = state.get(container_name)
            if not isinstance(container, dict):
                continue
            difficulty = difficulty or str(container.get("difficulty") or "")
            hints.append(str(container.get("optimization_hints") or ""))
        hint_text = " ".join(hints).lower()
        high_precision = difficulty.upper() == "L5" or (
            "c4" in hint_text and "precision" in hint_text
        )
        return (1e-6, 1e-9) if high_precision else (1e-3, 1e-6)

    @staticmethod
    def _determine_time_range(
        ode_system: dict,
        is_multi_pathway: bool,
        explicit_duration: float | None = None,
    ) -> tuple[float, float]:
        """确定仿真时间范围。

        多通路场景使用更长时间（120 min），单通路默认 60 min。
        若 ode_system 中有 dde_delay_minutes > 0，延长 t_end 以覆盖延迟效应。
        """
        t_start = 0.0
        t_end = explicit_duration or (120.0 if is_multi_pathway else 60.0)

        # 从 ode_system 提取时间信息
        if ode_system:
            dde_delay = float(ode_system.get("dde_delay_minutes", 0.0) or 0.0)
            if dde_delay > 0:
                # DDE 场景：t_end 至少为延迟的 4 倍（以观察多个振荡周期）
                t_end = max(t_end, dde_delay * 4.0)

        return t_start, t_end

    @staticmethod
    def _build_time_scales(is_multi_pathway: bool, t_end: float) -> list[dict[str, Any]]:
        """构建时间尺度列表。

        多通路场景输出多时间尺度（fast + slow），
        单通路输出单一时间尺度。
        """
        if is_multi_pathway:
            return [
                {
                    "name": "fast",
                    "t_start": 0.0,
                    "t_end": min(1.0, t_end),
                    "unit": "min",
                    "description": "快速过程：磷酸化、结合、cleavage",
                },
                {
                    "name": "slow",
                    "t_start": 0.0,
                    "t_end": t_end,
                    "unit": "min",
                    "description": "慢速过程：转录、翻译、降解",
                },
            ]
        return [
            {
                "name": "default",
                "t_start": 0.0,
                "t_end": t_end,
                "unit": "min",
                "description": "单一时间尺度",
            },
        ]

    @staticmethod
    def _default_plan() -> dict[str, Any]:
        """返回默认仿真计划（失败降级用）。"""
        return {
            "simulation_type": "ode",
            "solver": "scipy.solve_ivp",
            "t_start": 0.0,
            "t_end": 60.0,
            "n_points": 200,
            "time_scales": [
                {
                    "name": "default",
                    "t_start": 0.0,
                    "t_end": 60.0,
                    "unit": "min",
                    "description": "单一时间尺度（降级默认）",
                },
            ],
            "multi_pathway": False,
            "rtol": 1e-3,
            "atol": 1e-6,
        }


# =============================================================================
# DynamicRouter 兼容别名
# =============================================================================
# DynamicRouter._get_class_name 期望短名 SimulationPlanner（agent_registry_v4 约定）
SimulationPlanner = SimulationPlannerAgent


__all__ = [
    "SimulationPlannerAgent",
    "SimulationPlanner",
]
