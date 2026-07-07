# BioDynamics Agent v4 - Pathway Class Dispatcher（Phase 6 / Task 6.5.3）
#
# 基于 v4_pathway_class 分支：单通路 → 1 Specialist；多通路 → N Specialist + Coordinator。
#
# 设计原则（铁律）：
# 1. 不修改 v3 任何字段；不修改 P4 PathwayGraph（只读）
# 2. 不生成 ODE / 不做 SBML 验证（职责边界严格）
# 3. 失败降级：任何异常都返回空列表，不阻塞主流水线
# 4. Specialist 实例通过 pathway_registry.get_specialist() 获取，未注册返回 None 跳过
# 5. 多通路场景（len > 1）才调用 Cross-talk Coordinator
#
# v4_pathway_class 格式兼容（与 pathway_planner.parse_pathway_class 对齐）：
# - 单通路："EGFR_RTK"
# - 多通路："MULTI:EGFR_RTK+PI3K_AKT_mTOR"（pathway_planner 默认格式）
# - 兼容分隔符：";" / "," / "+"（部分上游可能使用其他分隔符）
#
# 参考：
# - spec.md Part 6 Dynamic Router pathway_class 分支（第 393-398 行）
# - tasks.md SubTask 6.5.3

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# PathwayClassDispatcher 主类
# =============================================================================
class PathwayClassDispatcher:
    """基于 v4_pathway_class 的通路分发器。

    职责：
    1. 解析 v4_pathway_class 字符串为 pathway 列表
    2. 单通路 → 调用 1 个 Specialist.apply_core()
    3. 多通路 → 调用 N 个 Specialist.apply_core() + Cross-talk Coordinator
    4. 返回 Specialist 调度结果列表

    不职责（铁律）：
    - 不修改 PathwayGraph（只读）
    - 不生成 ODE / 不做 SBML 验证
    - 不修改 Specialist 内部 Reaction（仅消费 apply_core 输出）

    用法::

        dispatcher = PathwayClassDispatcher()
        results = dispatcher.dispatch_specialists(state)
        # results = [
        #     {"pathway_class": "EGFR_RTK", "specialist_name": "...",
        #      "core_reactions": [...], "applied": True},
        #     ...
        # ]
    """

    def __init__(self) -> None:
        """初始化分发器。

        pathway_registry 通过懒加载获取（避免循环依赖与启动开销）。
        Cross-talk Coordinator 同样懒加载。
        """
        # pathway_registry 引用（懒加载，首次 dispatch_specialists 时填充）
        self._pathway_registry = None
        # 标记是否已尝试导入 specialist 子模块（避免每次 dispatch 都重试）
        self._specialists_imported: bool = False

    # -------------------------------------------------------------------------
    # pathway_class 解析
    # -------------------------------------------------------------------------
    def parse_pathway_class(self, v4_pathway_class: str) -> list[str]:
        """从 v4_pathway_class 字符串解析出 pathway 列表。

        支持以下格式（兼容多种上游格式）：
        - ``"EGFR_RTK"`` → ``["EGFR_RTK"]``（单通路）
        - ``"MULTI:EGFR_RTK+PI3K_AKT_mTOR"`` → ``["EGFR_RTK", "PI3K_AKT_mTOR"]``
        - ``"EGFR_RTK;PI3K_AKT_mTOR"`` → ``["EGFR_RTK", "PI3K_AKT_mTOR"]``
        - ``"EGFR_RTK,PI3K_AKT_mTOR"`` → ``["EGFR_RTK", "PI3K_AKT_mTOR"]``
        - ``""`` / ``"UNKNOWN"`` → ``[]``

        Args:
            v4_pathway_class: 通路类别字符串

        Returns:
            pathway 字符串列表（保持原顺序，去空）；UNKNOWN/空返回空列表
        """
        if not v4_pathway_class:
            return []
        s = v4_pathway_class.strip()
        if not s or s == "UNKNOWN":
            return []

        # 剥离 MULTI: 前缀（pathway_planner._format_multi_pathway 格式）
        if s.startswith("MULTI:"):
            s = s[len("MULTI:") :]

        # 统一替换分隔符为 "+" 后 split
        # 兼容 + / ; / ,（优先 + ，因为 pathway_planner 用 +）
        s = s.replace(";", "+").replace(",", "+")
        parts = [p.strip() for p in s.split("+")]
        # 去空 + 去重保持顺序
        seen: set[str] = set()
        result: list[str] = []
        for p in parts:
            if p and p != "UNKNOWN" and p not in seen:
                result.append(p)
                seen.add(p)
        return result

    def is_multi_pathway(self, v4_pathway_class: str) -> bool:
        """判断 v4_pathway_class 是否为多通路。

        判断依据：字符串中包含 ``";"`` / ``","`` / ``"+"`` 或 ``"MULTI:"`` 前缀。

        Args:
            v4_pathway_class: 通路类别字符串

        Returns:
            True 表示多通路；False 表示单通路或空/UNKNOWN
        """
        if not v4_pathway_class:
            return False
        s = v4_pathway_class.strip()
        if not s or s == "UNKNOWN":
            return False
        # 检查多通路标识
        if s.startswith("MULTI:"):
            return True
        # 检查分隔符
        return any(sep in s for sep in (";", ",", "+"))

    # -------------------------------------------------------------------------
    # Specialist 调度
    # -------------------------------------------------------------------------
    def dispatch_specialists(self, state: dict) -> list[dict]:
        """根据 v4_pathway_class 调度 Specialist，返回调度结果列表。

        行为：
        1. 从 state 读取 v4_pathway_class（缺失或 UNKNOWN → 返回空列表）
        2. 解析为 pathway 列表
        3. 对每个 pathway：
           - 通过 pathway_registry.get_specialist(pathway) 获取 Specialist 实例
           - 调用 specialist.apply_core({}) 获取 Reaction IR 片段
           - 记录结果 dict（含 pathway_class / specialist_name / core_reactions / applied）
        4. 多通路场景：额外调用 Cross-talk Coordinator
        5. 任何异常都返回已收集的部分结果（不抛出）

        Args:
            state: LangGraph 全局状态，读取 ``v4_pathway_class``

        Returns:
            Specialist 调度结果列表，每条 dict 含：
            - ``pathway_class``: str（通路类别键）
            - ``specialist_name``: str（Specialist 类名，未注册时为空）
            - ``core_reactions``: list（apply_core 输出的 reactions 字段）
            - ``applied``: bool（是否成功调用 apply_core）
            - ``error``: str|None（失败原因，成功时为 None）
        """
        try:
            pathway_class = state.get("v4_pathway_class", "") or ""
            pathways = self.parse_pathway_class(pathway_class)

            if not pathways:
                logger.debug(
                    "PathwayClassDispatcher: v4_pathway_class=%r 无可调度的通路",
                    pathway_class,
                )
                return []

            # 确保 10 个 Specialist 子模块已导入（触发 @register_specialist）
            self._ensure_specialists_imported()

            registry = self._get_pathway_registry()
            if registry is None:
                logger.warning(
                    "PathwayClassDispatcher: pathway_registry 不可用，"
                    "返回空 Specialist 调度列表"
                )
                return []

            results: list[dict] = []
            for pwc in pathways:
                result = self._dispatch_single_specialist(registry, pwc)
                results.append(result)

            # 多通路场景：调用 Cross-talk Coordinator
            if len(pathways) > 1:
                coordinator_result = self._dispatch_crosstalk_coordinator(
                    state, results, pathway_class
                )
                if coordinator_result is not None:
                    results.append(coordinator_result)

            logger.info(
                "PathwayClassDispatcher: pathway_class=%s, 调度 %d 个 Specialist",
                pathway_class,
                len(results),
            )
            return results
        except Exception as exc:
            logger.warning(
                "PathwayClassDispatcher.dispatch_specialists 失败，降级返回空列表: %s",
                exc,
            )
            return []

    # -------------------------------------------------------------------------
    # 内部辅助方法
    # -------------------------------------------------------------------------
    def _dispatch_single_specialist(
        self,
        registry: Any,
        pathway_class: str,
    ) -> dict:
        """调度单个 Specialist，返回调度结果 dict。

        Args:
            registry: pathway_registry 模块（含 get_specialist 函数）
            pathway_class: 通路类别键

        Returns:
            调度结果 dict（结构见 dispatch_specialists docstring）
        """
        try:
            specialist = registry.get_specialist(pathway_class)
            if specialist is None:
                logger.debug(
                    "PathwayClassDispatcher: pathway_class=%s 未注册 Specialist",
                    pathway_class,
                )
                return {
                    "pathway_class": pathway_class,
                    "specialist_name": "",
                    "core_reactions": [],
                    "applied": False,
                    "error": f"Specialist for {pathway_class} not registered",
                }

            # 调用 apply_core（pathway_graph 传空 dict，ontology_entities 默认 None）
            core_output = specialist.apply_core({})
            if not isinstance(core_output, dict):
                core_output = {}
            core_reactions = core_output.get("reactions", []) or []

            specialist_name = type(specialist).__name__
            logger.debug(
                "PathwayClassDispatcher: pathway_class=%s, specialist=%s, "
                "reactions=%d",
                pathway_class,
                specialist_name,
                len(core_reactions),
            )

            return {
                "pathway_class": pathway_class,
                "specialist_name": specialist_name,
                "core_reactions": core_reactions,
                "applied": True,
                "error": None,
            }
        except Exception as exc:
            logger.warning(
                "PathwayClassDispatcher: 调度 pathway_class=%s 的 Specialist 失败: %s",
                pathway_class,
                exc,
            )
            return {
                "pathway_class": pathway_class,
                "specialist_name": "",
                "core_reactions": [],
                "applied": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _dispatch_crosstalk_coordinator(
        self,
        state: dict,
        specialist_results: list[dict],
        pathway_class: str,
    ) -> dict | None:
        """多通路场景下调用 Cross-talk Coordinator。

        将 Specialist 调度结果转换为 Coordinator 期望的 specialist_outputs 格式，
        调用 CrossTalkCoordinator.coordinate()，返回协调结果 dict。

        Args:
            state: LangGraph 全局状态
            specialist_results: 已收集的 Specialist 调度结果列表
            pathway_class: 完整 v4_pathway_class 字符串

        Returns:
            Coordinator 调度结果 dict（含 coordinator: True 标记）；失败返回 None
        """
        try:
            # 懒加载 Coordinator（避免循环依赖）
            from app.crosstalk.coordinator import CrossTalkCoordinator

            coordinator = CrossTalkCoordinator()

            # 将 Specialist 调度结果转换为 Coordinator 期望的输入格式
            specialist_outputs: list[dict] = []
            for res in specialist_results:
                if not res.get("applied"):
                    continue
                specialist_outputs.append(
                    {
                        "pathway_class": res.get("pathway_class", ""),
                        "reactions": res.get("core_reactions", []),
                        "species": [],
                    }
                )

            coord_result = coordinator.coordinate(
                specialist_outputs=specialist_outputs,
                pathway_class=pathway_class,
            )

            return {
                "pathway_class": pathway_class,
                "specialist_name": "CrossTalkCoordinator",
                "core_reactions": [],
                "applied": True,
                "coordinator": True,
                "coordinator_output": coord_result,
                "error": None,
            }
        except Exception as exc:
            logger.warning(
                "PathwayClassDispatcher: Cross-talk Coordinator 调度失败: %s",
                exc,
            )
            return {
                "pathway_class": pathway_class,
                "specialist_name": "CrossTalkCoordinator",
                "core_reactions": [],
                "applied": False,
                "coordinator": True,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _get_pathway_registry(self) -> Any:
        """懒加载 pathway_registry 模块（避免循环依赖）。

        Returns:
            pathway_registry 模块；不可用时返回 None
        """
        if self._pathway_registry is not None:
            return self._pathway_registry
        try:
            from app.pathways import pathway_registry

            self._pathway_registry = pathway_registry
            return pathway_registry
        except Exception as exc:
            logger.warning(
                "PathwayClassDispatcher: 导入 pathway_registry 失败: %s",
                exc,
            )
            return None

    def _ensure_specialists_imported(self) -> None:
        """懒加载 10 个 Specialist 子模块，触发 @register_specialist 注册。

        首次调用时执行 import，后续调用直接返回（_specialists_imported 标记）。
        任一 Specialist 导入失败时记录 warning 但不阻塞（部分注册仍可用）。
        """
        if self._specialists_imported:
            return
        self._specialists_imported = True
        _specialist_modules = [
            "app.pathways.specialists.egfr_specialist",
            "app.pathways.specialists.mapk_specialist",
            "app.pathways.specialists.pi3k_akt_mtor_specialist",
            "app.pathways.specialists.p53_specialist",
            "app.pathways.specialists.apoptosis_specialist",
            "app.pathways.specialists.cell_cycle_specialist",
            "app.pathways.specialists.jak_stat_specialist",
            "app.pathways.specialists.nf_kappa_b_specialist",
            "app.pathways.specialists.wnt_specialist",
            "app.pathways.specialists.tgf_beta_specialist",
        ]
        import importlib

        for mod_name in _specialist_modules:
            try:
                importlib.import_module(mod_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PathwayClassDispatcher: 导入 %s 失败: %s",
                    mod_name,
                    exc,
                )


__all__ = ["PathwayClassDispatcher"]
