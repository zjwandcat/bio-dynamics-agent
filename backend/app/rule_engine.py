"""BioDynamics Agent - 规则引擎（v2 升级）

职责：
- 校验 ODE 模板选择、参数范围、单位、激活/抑制方向、Hill 系数、初值。
- 任何 *error* 级违例 → 触发 LangGraph 重试，不交由 LLM 判定。
- 任何 *warning* 级违例 → 继续生成，但在 `ode_model.rule_violations` 中记录。

设计原则：
- 各 Rule 类实现 `check(network_relations, parameters) -> list[Violation]`。
- `RuleEngine` 顺序执行所有 Rule，合并 Violation 列表。
- 不依赖 LangChain / LangGraph，pure Python，便于单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable


# -----------------------------------------------------------------------------
# 数据结构
# -----------------------------------------------------------------------------
@dataclass
class Violation:
    """单条规则违例。"""

    rule_name: str
    edge_key: str | None
    message: str
    severity: str  # "error" | "warning"


@dataclass
class RuleResult:
    """规则引擎执行结果。"""

    ok: bool
    violations: list[Violation] = field(default_factory=list)

    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warning"]


# -----------------------------------------------------------------------------
# Rule 协议
# -----------------------------------------------------------------------------
@runtime_checkable
class Rule(Protocol):
    """规则协议。实现者只需提供 `name` 和 `check` 两个成员。"""

    name: str

    def check(
        self,
        network_relations: dict,
        parameters: dict,
    ) -> list[Violation]: ...


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def _edges(relations: dict) -> list[dict]:
    """从 LLM 输出的 network_relations 中安全提取边列表。"""
    if not isinstance(relations, dict):
        return []
    return list(relations.get("relations") or [])


def _edge_key(edge: dict) -> str:
    """统一的 edge_key 表示。"""
    if edge.get("edge_key"):
        return str(edge["edge_key"])
    src = edge.get("source", "")
    tgt = edge.get("target", "")
    return f"{src}|{tgt}"


def _param_for_edge(parameters: dict, edge_key: str) -> dict:
    """从 parameters 字典中提取某条边的参数。"""
    if not isinstance(parameters, dict):
        return {}
    return parameters.get(edge_key) or {}


# -----------------------------------------------------------------------------
# 规则 1：模板槽位覆盖
# -----------------------------------------------------------------------------
class TemplateRule:
    """确保每条边都有可用的模板槽位。"""

    name: str = "template"

    def check(self, network_relations: dict, parameters: dict) -> list[Violation]:
        violations: list[Violation] = []
        for edge in _edges(network_relations):
            key = _edge_key(edge)
            if not edge.get("interaction") in ("activation", "inhibition", "conversion"):
                violations.append(
                    Violation(
                        rule_name=self.name,
                        edge_key=key,
                        message=f"interaction 取值非法：{edge.get('interaction')!r}",
                        severity="error",
                    )
                )
        return violations


# -----------------------------------------------------------------------------
# 规则 2：参数范围
# -----------------------------------------------------------------------------
class ParameterRangeRule:
    """常见动力学参数的合理区间。"""

    name: str = "parameter_range"
    # Kd / Km / EC50 / IC50 单位统一为 nM；half-life / rate 统一为 h
    RANGES: dict[str, tuple[float, float]] = {
        "kd": (1e-5, 1e6),
        "km": (1e-5, 1e6),
        "ec50": (1e-5, 1e6),
        "ic50": (1e-5, 1e6),
        "vmax": (1e-6, 1e6),
        "half_life": (0.01, 1000.0),
        "half-life": (0.01, 1000.0),
        "degradation": (1e-6, 100.0),
        "secretion": (1e-6, 100.0),
    }

    def check(self, network_relations: dict, parameters: dict) -> list[Violation]:
        violations: list[Violation] = []
        for key, params in (parameters or {}).items():
            if not isinstance(params, dict):
                continue
            for pname, info in params.items():
                if pname in ("param_found", "fallback_to_estimation", "reasoning", "edge_key"):
                    continue
                if not isinstance(info, dict):
                    continue
                val = info.get("value")
                if val is None:
                    continue
                bounds = self.RANGES.get(str(pname).lower())
                if bounds is None:
                    continue
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    violations.append(
                        Violation(
                            rule_name=self.name,
                            edge_key=str(key),
                            message=f"{pname} 非数值：{val!r}",
                            severity="error",
                        )
                    )
                    continue
                lo, hi = bounds
                if v < lo or v > hi:
                    violations.append(
                        Violation(
                            rule_name=self.name,
                            edge_key=str(key),
                            message=(
                                f"{pname}={v} 超出允许范围 [{lo}, {hi}]"
                            ),
                            severity="warning",
                        )
                    )
        return violations


# -----------------------------------------------------------------------------
# 规则 3：单位
# -----------------------------------------------------------------------------
class UnitRule:
    """参数单位必须落在已知的白名单内。"""

    name: str = "unit"
    ALLOWED: set[str] = {"nM", "h", "s", "M", "mM", "uM", "min", ""}

    def check(self, network_relations: dict, parameters: dict) -> list[Violation]:
        violations: list[Violation] = []
        for key, params in (parameters or {}).items():
            if not isinstance(params, dict):
                continue
            for pname, info in params.items():
                if pname in ("param_found", "fallback_to_estimation", "reasoning", "edge_key"):
                    continue
                if not isinstance(info, dict):
                    continue
                unit = str(info.get("unit", "")).strip()
                if unit not in self.ALLOWED:
                    violations.append(
                        Violation(
                            rule_name=self.name,
                            edge_key=str(key),
                            message=f"{pname} 单位未识别：{unit!r}",
                            severity="warning",
                        )
                    )
        return violations


# -----------------------------------------------------------------------------
# 规则 4：激活/抑制方向
# -----------------------------------------------------------------------------
class ActivationDirectionRule:
    """抑制边应配 Kd；激活边应配 Vmax（缺一即告警）。"""

    name: str = "activation_direction"

    def check(self, network_relations: dict, parameters: dict) -> list[Violation]:
        violations: list[Violation] = []
        for edge in _edges(network_relations):
            key = _edge_key(edge)
            interaction = edge.get("interaction", "")
            params = _param_for_edge(parameters, key)
            if interaction == "inhibition":
                if not any(str(k).lower().startswith("kd") for k in params.keys()):
                    violations.append(
                        Violation(
                            rule_name=self.name,
                            edge_key=key,
                            message="抑制边缺少 Kd 参数",
                            severity="warning",
                        )
                    )
            elif interaction == "activation":
                if not any(
                    str(k).lower().startswith("vmax") or "vmax" in str(k).lower()
                    for k in params.keys()
                ):
                    violations.append(
                        Violation(
                            rule_name=self.name,
                            edge_key=key,
                            message="激活边缺少 Vmax 参数",
                            severity="warning",
                        )
                    )
        return violations


# -----------------------------------------------------------------------------
# 规则 5：Hill 系数
# -----------------------------------------------------------------------------
class HillCoefficientRule:
    """Hill 系数 n ∈ [0.5, 6]；超出范围告警。"""

    name: str = "hill_coefficient"

    def check(self, network_relations: dict, parameters: dict) -> list[Violation]:
        violations: list[Violation] = []
        for edge in _edges(network_relations):
            key = _edge_key(edge)
            n = edge.get("hill_coefficient")
            if n is None:
                continue
            try:
                n_val = float(n)
            except (TypeError, ValueError):
                violations.append(
                    Violation(
                        rule_name=self.name,
                        edge_key=key,
                        message=f"hill_coefficient 非数值：{n!r}",
                        severity="error",
                    )
                )
                continue
            if n_val < 0.5 or n_val > 6.0:
                violations.append(
                    Violation(
                        rule_name=self.name,
                        edge_key=key,
                        message=f"Hill 系数 n={n_val} 超出范围 [0.5, 6]",
                        severity="warning",
                    )
                )
        return violations


# -----------------------------------------------------------------------------
# 规则 6：初值
# -----------------------------------------------------------------------------
class InitialValueRule:
    """y0[i] ≥ 0 且 ≤ 1e6。"""

    name: str = "initial_value"
    MIN: float = 0.0
    MAX: float = 1e6

    def check(self, network_relations: dict, parameters: dict) -> list[Violation]:
        violations: list[Violation] = []
        relations = network_relations if isinstance(network_relations, dict) else {}
        y0 = relations.get("y0") or {}
        if not isinstance(y0, dict):
            return violations
        for sp, v in y0.items():
            try:
                v_val = float(v)
            except (TypeError, ValueError):
                violations.append(
                    Violation(
                        rule_name=self.name,
                        edge_key=None,
                        message=f"物种 {sp} 初值非数值：{v!r}",
                        severity="error",
                    )
                )
                continue
            if v_val < self.MIN or v_val > self.MAX:
                violations.append(
                    Violation(
                        rule_name=self.name,
                        edge_key=None,
                        message=f"物种 {sp} 初值 {v_val} 超出范围 [{self.MIN}, {self.MAX}]",
                        severity="error",
                    )
                )
        return violations


# -----------------------------------------------------------------------------
# 规则引擎
# -----------------------------------------------------------------------------
class RuleEngine:
    """顺序执行多条 Rule，合并 Violation 列表。"""

    DEFAULT_RULES: list[Rule] = [
        TemplateRule(),
        ParameterRangeRule(),
        UnitRule(),
        ActivationDirectionRule(),
        HillCoefficientRule(),
        InitialValueRule(),
    ]

    def __init__(self, rules: Iterable[Rule] | None = None) -> None:
        self.rules: list[Rule] = list(rules) if rules is not None else list(self.DEFAULT_RULES)

    def check(self, network_relations: dict, parameters: dict) -> RuleResult:
        """对 (network_relations, parameters) 执行所有规则。"""
        all_violations: list[Violation] = []
        for rule in self.rules:
            try:
                all_violations.extend(rule.check(network_relations, parameters))
            except Exception as exc:  # 单条规则异常不阻塞其他规则
                all_violations.append(
                    Violation(
                        rule_name=getattr(rule, "name", "unknown"),
                        edge_key=None,
                        message=f"规则执行异常：{exc}",
                        severity="warning",
                    )
                )
        ok = len([v for v in all_violations if v.severity == "error"]) == 0
        return RuleResult(ok=ok, violations=all_violations)
