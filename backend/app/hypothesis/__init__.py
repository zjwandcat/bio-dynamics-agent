# BioDynamics Agent v4 - Hypothesis Layer（Phase 6 / Task 6.1）
#
# 假设层包：Hypothesis Agent + 子组件（Generator / Experiment Designer /
# Falsification Checker / Parameter Explorer / Sensitivity Planner）。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_HYPOTHESIS_AGENT_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段；仅消费 v4_ode_system / metrics / v4_validation_report /
#    v4_grounding_ledger / v4_sensitivity_report
# 3. 失败降级：任何异常都返回空列表，不阻塞报告生成
# 4. 假设生成失败 → 输出空 v4_hypothesis_list + warning
# 5. 与 Validation 交互：Hypothesis Agent 仅在 v4_validation_report.overall_pass=True
#    时执行（Validation 失败短路到错误报告，不进入 Hypothesis）
#
# 对应 spec.md Part 5（第 350-398 行）

from app.hypothesis.hypothesis_agent import (
    HypothesisAgent,
    hypothesis_agent_hook_node,
)
from app.hypothesis.hypothesis_generator import (
    HypothesisGenerator,
    HypothesisStrategy,
)
from app.hypothesis.experiment_designer import ExperimentDesigner
from app.hypothesis.falsifiability_checker import FalsificationChecker
from app.hypothesis.parameter_explorer import ParameterExplorer
from app.hypothesis.sensitivity_planner import (
    SensitivityPlanner,
    METHOD_LOCAL,
    METHOD_MORRIS,
    METHOD_SOBOL,
)

__all__ = [
    "HypothesisAgent",
    "hypothesis_agent_hook_node",
    "HypothesisGenerator",
    "HypothesisStrategy",
    "ExperimentDesigner",
    "FalsificationChecker",
    "ParameterExplorer",
    "SensitivityPlanner",
    "METHOD_LOCAL",
    "METHOD_MORRIS",
    "METHOD_SOBOL",
]
