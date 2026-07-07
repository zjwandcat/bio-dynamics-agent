# BioDynamics Agent v4 - Parameter Agent（Phase 6 / Task 6.6.4）
#
# 强制 pathway_tag 隔离 + threshold 验证 + provenance 溯源。
#
# 职责：
# 1. 遍历 v3 parameters，为每个参数标注 pathway_tag（来源通路隔离）
# 2. 验证参数阈值（k_cat > 0, Km > 0, 速率常数 > 0）
# 3. 记录参数 provenance（RAG/SBML/calibration/estimation/literature）
# 4. pathway_tag 隔离：通路 A 的参数不得泄漏到通路 B 的反应
# 5. 合并 v4_calibration_result 的校准参数（优先级最高）
#
# 设计原则（铁律）：
# 1. Feature Flag V4_DYNAMIC_ROUTING_ENABLED=false → 返回 {}（不执行）
# 2. 不修改 v3 parameters；仅新增 v4_parameter_registry
# 3. 失败降级：任何异常返回 {"v4_parameter_registry": {"params": [], ...}}
# 4. pathway_tag 隔离：每个参数必须标注 pathway_tag，跨通路参数标记为 isolation 违规
# 5. threshold 验证：动力学参数（k_cat/Km/k*/Vmax）必须 > 0
#
# 参考：
# - app.state.BioDynamicsState.parameters（v3 参数字段）
# - app.calibration.calibration_agent.CalibrationAgent（v4 校准结果）
# - tasks.md SubTask 6.6.4

from __future__ import annotations

import logging
from typing import Any

# app.config 无循环依赖风险（不导入 agents_v4），可在模块级导入
from app.config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# 常量
# =============================================================================
# 必须为正值的参数名前缀/全名（threshold 验证用）
# k_cat / kcat / Km / km / Vmax / k_on / k_off / k1 / k2 / k3 等速率常数必须 > 0
_POSITIVE_REQUIRED_KEYWORDS: tuple[str, ...] = (
    "k_cat", "kcat", "km_", "_km", "vmax", "k_on", "koff", "k_off",
    "kon", "k1", "k2", "k3", "k4", "k_d", "kd", "n_", "hill",
)

# 参数来源优先级（数值越小优先级越高）
_SOURCE_PRIORITY: dict[str, int] = {
    "calibration": 0,
    "RAG": 1,
    "rag": 1,
    "SBML": 2,
    "sbml": 2,
    "PubMed": 3,
    "pubmed": 3,
    "literature": 3,
    "KEGG": 4,
    "kegg": 4,
    "UniProt": 4,
    "uniprot": 4,
    "Inferred": 9,
    "inferred": 9,
    "estimation": 9,
    "unknown": 99,
}


class ParameterAgent:
    """v4 参数管理 Agent：强制 pathway_tag 隔离 + threshold + provenance。

    遍历 v3 parameters，为每个参数标注 pathway_tag、验证阈值、记录溯源，
    并合并 v4_calibration_result 的校准参数。强制通路间参数隔离。

    用法::

        agent = ParameterAgent()
        update = agent.manage(state)
        # update = {"v4_parameter_registry": {params, isolation_valid, warnings}}
    """

    AGENT_VERSION: str = "v4.0"

    # -------------------------------------------------------------------------
    # 主入口
    # -------------------------------------------------------------------------
    def manage(self, state: dict) -> dict:
        """主入口：管理参数 pathway_tag 隔离 + threshold + provenance。

        Args:
            state: LangGraph 全局状态，读取：
                - ``parameters``: v3 参数 dict（{edge_key: {param_name: value}} 或扁平 dict）
                - ``v4_pathway_class``: 通路类别字符串
                - ``v4_calibration_result``: v4 校准结果（含 calibrated_params）
                - ``v4_pathway_graph``: PathwayGraph（用于 edge_key → pathway_tag 映射）

        Returns:
            flag=false 时返回 {}
            正常时返回 ``{"v4_parameter_registry": {...}}``
            失败时返回降级 registry

            v4_parameter_registry 结构::

                {
                    "params": [
                        {
                            "name": str,
                            "value": float,
                            "pathway_tag": str,
                            "source": str,          # provenance
                            "threshold_valid": bool,
                        },
                        ...
                    ],
                    "isolation_valid": bool,
                    "warnings": list[str],
                }
        """
        # 1. Feature Flag 检查（铁律：flag=false 不执行）
        if not getattr(settings, "V4_DYNAMIC_ROUTING_ENABLED", False):
            logger.debug("V4_DYNAMIC_ROUTING_ENABLED=false，ParameterAgent 跳过")
            return {}

        try:
            # 2. 提取输入
            parameters = state.get("parameters") or {}
            pathway_class = state.get("v4_pathway_class", "") or ""
            calibration_result = state.get("v4_calibration_result") or {}
            pathway_graph = state.get("v4_pathway_graph") or {}

            # 3. 构建 edge_key → pathway_tag 映射（从 pathway_graph edges）
            edge_tag_map = self._build_edge_tag_map(pathway_graph)

            # 4. 归一化参数为统一列表
            raw_params = self._normalize_parameters(parameters, pathway_class, edge_tag_map)

            # 5. 合并校准结果（优先级最高）
            calibrated_params = self._extract_calibrated_params(calibration_result, pathway_class)
            raw_params = self._merge_calibrated(raw_params, calibrated_params)

            # 6. 验证 threshold + 标注 provenance
            param_entries: list[dict[str, Any]] = []
            warnings: list[str] = []

            for p in raw_params:
                name = p.get("name", "")
                value = p.get("value")
                pathway_tag = p.get("pathway_tag", "") or pathway_class
                source = p.get("source", "unknown")

                # threshold 验证
                threshold_valid = self._validate_threshold(name, value)
                if not threshold_valid:
                    warnings.append(
                        f"参数 '{name}' 阈值验证失败：value={value}（速率常数/Km 必须 > 0）"
                    )

                # provenance 验证
                if source == "unknown":
                    warnings.append(f"参数 '{name}' 缺少 provenance 溯源信息")

                param_entries.append({
                    "name": name,
                    "value": value,
                    "pathway_tag": pathway_tag,
                    "source": source,
                    "threshold_valid": threshold_valid,
                })

            # 7. pathway_tag 隔离验证
            isolation_valid, isolation_warnings = self._check_isolation(
                param_entries, pathway_class
            )
            warnings.extend(isolation_warnings)

            logger.info(
                "ParameterAgent: 管理 %d 个参数，isolation_valid=%s，%d 条警告",
                len(param_entries), isolation_valid, len(warnings),
            )

            return {
                "v4_parameter_registry": {
                    "params": param_entries,
                    "isolation_valid": isolation_valid,
                    "warnings": warnings,
                }
            }

        except Exception as exc:
            # 失败降级：返回空 registry，不阻塞流水线
            logger.warning("ParameterAgent.manage 失败，降级返回空 registry: %s", exc)
            return {
                "v4_parameter_registry": {
                    "params": [],
                    "isolation_valid": True,
                    "warnings": [f"ParameterAgent 管理失败: {exc}"],
                }
            }

    def generate(self, state: dict) -> dict:
        """DynamicRouter 调度入口（别名，委托给 manage）。"""
        return self.manage(state)

    # -------------------------------------------------------------------------
    # 内部辅助方法
    # -------------------------------------------------------------------------
    @staticmethod
    def _build_edge_tag_map(pathway_graph: dict) -> dict[str, str]:
        """从 pathway_graph.edges 构建 edge_id → pathway_tag 映射。

        用于将 v3 parameters 的 edge_key 映射到 pathway_tag。
        """
        edge_tag_map: dict[str, str] = {}
        edges = pathway_graph.get("edges", []) or []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            edge_id = edge.get("id", "") or edge.get("reaction_id", "")
            tag = edge.get("pathway_tag", "")
            if edge_id and tag:
                edge_tag_map[edge_id] = tag
        return edge_tag_map

    @staticmethod
    def _normalize_parameters(
        parameters: dict,
        pathway_class: str,
        edge_tag_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        """将 v3 parameters 归一化为统一的参数列表。

        支持多种 v3 parameters 格式：
        1. 扁平 dict：{param_name: numeric_value}
        2. 按 edge 分组：{edge_key: {param_name: numeric_value}}
        3. 详细格式：{edge_key: {param_name: {value, source, confidence, ...}}}
        """
        raw_params: list[dict[str, Any]] = []

        if not parameters or not isinstance(parameters, dict):
            return raw_params

        for key, val in parameters.items():
            if isinstance(val, (int, float)):
                # 格式 1：扁平 dict {param_name: value}
                raw_params.append({
                    "name": key,
                    "value": float(val),
                    "pathway_tag": pathway_class,
                    "source": "unknown",
                })
            elif isinstance(val, dict):
                # 格式 2/3：按 edge 分组
                edge_tag = edge_tag_map.get(key, pathway_class)
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, (int, float)):
                        raw_params.append({
                            "name": sub_key,
                            "value": float(sub_val),
                            "pathway_tag": edge_tag,
                            "source": "unknown",
                        })
                    elif isinstance(sub_val, dict):
                        # 格式 3：详细格式
                        raw_params.append({
                            "name": sub_key,
                            "value": sub_val.get("value"),
                            "pathway_tag": sub_val.get("pathway_tag", edge_tag),
                            "source": sub_val.get("source", "unknown"),
                        })

        return raw_params

    @staticmethod
    def _extract_calibrated_params(
        calibration_result: dict,
        pathway_class: str,
    ) -> list[dict[str, Any]]:
        """从 v4_calibration_result 提取校准参数。

        校准参数的 source 标注为 "calibration"（优先级最高）。
        """
        result: list[dict[str, Any]] = []
        if not calibration_result or not isinstance(calibration_result, dict):
            return result

        calibrated = calibration_result.get("calibrated_params") or {}
        if not isinstance(calibrated, dict):
            return result

        for name, value in calibrated.items():
            if isinstance(value, (int, float)):
                result.append({
                    "name": name,
                    "value": float(value),
                    "pathway_tag": pathway_class,
                    "source": "calibration",
                })
            elif isinstance(value, dict):
                result.append({
                    "name": name,
                    "value": value.get("value"),
                    "pathway_tag": value.get("pathway_tag", pathway_class),
                    "source": "calibration",
                })

        return result

    @staticmethod
    def _merge_calibrated(
        raw_params: list[dict[str, Any]],
        calibrated_params: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """合并校准参数（优先级最高，覆盖同名的非校准参数）。

        策略：
        1. 校准参数按 name 建立索引
        2. 遍历 raw_params，同名校准参数覆盖原值与 source
        3. 追加校准参数中新增的（raw_params 中不存在的）
        """
        if not calibrated_params:
            return raw_params

        # 建立校准参数索引
        cal_index: dict[str, dict[str, Any]] = {
            p["name"]: p for p in calibrated_params if "name" in p
        }

        merged: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        for p in raw_params:
            name = p.get("name", "")
            if name in cal_index:
                # 校准参数覆盖
                merged.append(cal_index[name])
                seen_names.add(name)
            else:
                merged.append(p)
                if name:
                    seen_names.add(name)

        # 追加校准参数中新增的
        for name, cal_p in cal_index.items():
            if name not in seen_names:
                merged.append(cal_p)

        return merged

    @staticmethod
    def _validate_threshold(name: str, value: Any) -> bool:
        """验证参数阈值（k_cat/Km/速率常数 必须 > 0）。

        规则：
        1. value 必须为数值类型
        2. 名为 k_cat/Km/Vmax/k1/k2/k_on/k_off 等的参数必须 > 0
        3. 其他参数 >= 0（允许 0 值，如某些初始浓度）
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False

        n = name.lower()
        # 检查是否为必须正值的参数
        is_positive_required = False
        for kw in _POSITIVE_REQUIRED_KEYWORDS:
            if kw in n:
                is_positive_required = True
                break
        # k 开头的单字母+数字（如 k1, k2, k3）也要求 > 0
        if n.startswith("k") and len(n) <= 3:
            is_positive_required = True

        if is_positive_required:
            return v > 0
        return v >= 0

    @staticmethod
    def _check_isolation(
        param_entries: list[dict[str, Any]],
        pathway_class: str,
    ) -> tuple[bool, list[str]]:
        """验证 pathway_tag 隔离。

        检查规则：
        1. 每个参数必须有 pathway_tag（非空）
        2. 参数的 pathway_tag 应与当前通路一致
        3. 若出现不同 pathway_tag 的参数，记录 warning（潜在跨通路泄漏）
        4. isolation_valid = True 当且仅当无跨通路泄漏

        Returns:
            (isolation_valid, warnings)
        """
        warnings: list[str] = []
        foreign_tags: set[str] = set()

        for p in param_entries:
            tag = p.get("pathway_tag", "")
            if not tag:
                warnings.append(
                    f"参数 '{p.get('name', '')}' 缺少 pathway_tag 标注"
                )
            elif pathway_class and tag != pathway_class:
                # 跨通路参数（潜在泄漏）
                foreign_tags.add(tag)

        for ftag in foreign_tags:
            warnings.append(
                f"pathway_tag 隔离警告：参数中存在来自通路 '{ftag}' 的参数，"
                f"当前通路为 '{pathway_class}'，请确认是否为跨通路共享参数"
            )

        # isolation_valid：无跨通路泄漏且无缺失 tag
        isolation_valid = len(foreign_tags) == 0 and not any(
            not p.get("pathway_tag") for p in param_entries
        )

        return isolation_valid, warnings


# =============================================================================
# DynamicRouter 兼容性说明
# =============================================================================
# DynamicRouter._get_class_name 对 "parameter_agent" 返回 "ParameterAgent"，
# 与本类名一致，无需额外别名。


__all__ = [
    "ParameterAgent",
]
