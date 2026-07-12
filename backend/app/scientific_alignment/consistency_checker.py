# BioDynamics Agent v4 - Scientific Alignment Loop: Consistency Checker (Task 24)
#
# 科学一致性检查器：从 Canonical Reference 加载自洽规则，对仿真结果做机制级逻辑校验。
#
# 设计背景：
#   当前 Validation 只检查数值合理性（如范围、收敛性），不会发现逻辑矛盾。
#   例如 "EGFR Peak 63min + ERK Peak 16min" 数值都合理，但违反机制因果顺序
#   （EGFR 是 ERK 上游，其 Peak 不可能晚于 ERK Peak）。Consistency Checker
#   从 Canonical 的 consistency_rules 段加载断言，对仿真 metrics 做机制级校验。
#
# 安全设计（参考 security-best-practices / canonical_loader.py）：
#   1. assertion 字符串来自 Canonical YAML，本质是外部数据，绝不可直接 eval/exec
#   2. 三层防护：
#      a. 字符白名单：仅允许标识符、数字、比较/逻辑运算符、括号、空白
#      b. AST 白名单：ast.parse + 节点类型 allowlist（Compare/BoolOp/UnaryOp/
#         Name/Constant/And/Or/Not/Gt/Lt/GtE/LtE/Eq/NotEq）
#      c. 受限命名空间：eval(expr, {"__builtins__": {}}, simulation_metrics)
#         ——仅在前两层通过后才执行，且 builtins 完全清空
#   3. 标识符缺失（metric 未提供）→ 标记 not_evaluated，不计入违规也不计入 passed
#   4. 求值异常 → 保守记为 violation（label="evaluation_error"），避免静默漏检
#
# 依赖：Python 标准库（ast/re/dataclasses）+ 已有 canonical_loader；不引入新依赖。
#
# 核心导出：
#   from app.scientific_alignment.consistency_checker import (
#       ConsistencyViolation, ConsistencyReport,
#       check_consistency, extract_peak_times_from_simulation,
#   )

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.scientific_alignment.canonical_loader import (
    CanonicalNotFoundError,
    ConsistencyRule,
    get_consistency_rules,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================
# 字符白名单正则：仅允许标识符字符、数字、小数点、比较运算符（< > = !）、
# 逻辑运算符（and/or/not 由字母组成，已被标识符字符覆盖）、括号、空白。
# 拒绝 ; : , [ ] { } @ $ % ^ & * + - / | ` ~ 等危险字符与算术运算符。
_ALLOWED_CHAR_PATTERN: re.Pattern[str] = re.compile(
    r"^[a-zA-Z0-9_\s\.\<\>\=\!\(\)]+$"
)

# 标识符提取正则：用于从 assertion 中提取变量名，检查是否在 simulation_metrics 中
_IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

# Python 关键字集合：and/or/not/True/False/None 等虽匹配标识符正则，但不是变量名，
# 不应要求出现在 simulation_metrics 中。其他关键字（if/for/import/...）会被
# AST 白名单拒绝（不允许 If/For/Import 等节点）。
_PY_KEYWORDS_AS_NAMES: frozenset[str] = frozenset(
    {"and", "or", "not", "True", "False", "None"}
)

# AST 节点白名单：仅允许以下节点类型出现在 assertion AST 中。
# 任何 Call/Attribute/Lambda/If/For/Import/List/Dict/Set/Subscript 等均被拒绝。
# 注意：ast.Load 是 ast.Name.ctx 的上下文标记（表示"读取变量"），是安全叶子节点，
#       必须加入白名单，否则 ast.walk 会遍历到它并误判为不安全。
_ALLOWED_AST_NODES: tuple[type, ...] = (
    ast.Expression,   # mode='eval' 顶层节点
    ast.BoolOp,       # and / or
    ast.UnaryOp,      # not（op 仅允许 Not）
    ast.Compare,      # 比较运算（ops 仅允许 Gt/Lt/GtE/LtE/Eq/NotEq）
    ast.Name,         # 变量名
    ast.Constant,     # 数字/布尔字面量
    ast.Load,         # Name.ctx 读取上下文（安全叶子节点）
    ast.And,
    ast.Or,
    ast.Not,
    ast.Gt,
    ast.Lt,
    ast.GtE,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
)

# 物种名 → 输出指标键 的归一化映射表
# 键为小写 + 去除 -/_ 后的规范形式，值为输出 metric 名
_SPECIES_TO_METRIC: Dict[str, str] = {
    # EGFR 类（peak time，取 argmax）
    "egfr": "egfr_peak_time",
    "pegfr": "egfr_peak_time",
    "p_egfr": "egfr_peak_time",
    "egfr_p": "egfr_peak_time",
    # ERK 类（peak time，取 argmax）
    "erk": "erk_peak_time",
    "perk": "erk_peak_time",
    "pperk": "erk_peak_time",
    "p_erk": "erk_peak_time",
    "pp_erk": "erk_peak_time",
    # RAS 类（activation time，取首超最大值 50% 的时刻）
    "ras": "ras_activation_time",
    "rasgtp": "ras_activation_time",
    "ras_gtp": "ras_activation_time",
    "ras_gtp_active": "ras_activation_time",
    # DUSP 类（expression time，取首超最大值 50% 的时刻）
    "dusp": "dusp_expression_time",
    "dusp1": "dusp_expression_time",
}

# "peak time" 语义指标集合：用 argmax 取峰值时刻
_PEAK_TIME_METRICS: frozenset[str] = frozenset(
    {"egfr_peak_time", "erk_peak_time"}
)

# "activation/expression time" 语义指标集合：取首个超过最大值 50% 的时刻，
# 更符合"活化时间/表达启动时间"的生物学语义（累积型响应的阈值穿越时刻）
_THRESHOLD_TIME_METRICS: frozenset[str] = frozenset(
    {"ras_activation_time", "dusp_expression_time"}
)

# 阈值比例：首超最大值的 50%（与规范一致）
_ACTIVATION_THRESHOLD_RATIO: float = 0.5


# =============================================================================
# 数据类
# =============================================================================
@dataclass
class ConsistencyViolation:
    """单条一致性违规记录。

    Attributes:
        rule: 规则的人类可读描述（来自 Canonical，如
            ``"EGFR Peak 不能晚于 ERK Peak"``）。
        assertion: 断言原文（如 ``"egfr_peak_time < erk_peak_time"``）。
        violation_label: 违规标签（来自 Canonical，如
            ``"egfr_peak_after_erk_peak"``）；求值异常时为 ``"evaluation_error"``。
        observed_values: 实际观测到的相关指标值 dict，便于排查根因。
        message: 人类可读的违规说明（用于日志与 UI 展示）。
    """

    rule: str = ""
    assertion: str = ""
    violation_label: str = ""
    observed_values: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class ConsistencyReport:
    """一致性检查报告。

    Attributes:
        pathway: 被检查的通路标识（如 ``"egfr"``）。
        passed: 是否通过一致性检查。仅当已评估规则中无任何违规时为 True；
            未评估的规则（标识符缺失）不影响 passed。
        violations: 违规列表（ConsistencyViolation）。
        rules_checked: 总规则数（从 Canonical 加载的规则数）。
        rules_evaluated: 已成功评估的规则数（不含 not_evaluated 与异常）。
    """

    pathway: str = ""
    passed: bool = True
    violations: List[ConsistencyViolation] = field(default_factory=list)
    rules_checked: int = 0
    rules_evaluated: int = 0


# =============================================================================
# 安全表达式求值器
# =============================================================================
def _extract_identifiers(assertion: str) -> List[str]:
    """从 assertion 中提取所有标识符（变量名候选）。

    Args:
        assertion: 断言字符串，如 ``"egfr_peak_time < erk_peak_time"``。

    Returns:
        去重后的标识符列表（保留出现顺序），如
        ``["egfr_peak_time", "erk_peak_time"]``。
    """
    seen: Dict[str, None] = {}
    for ident in _IDENTIFIER_PATTERN.findall(assertion):
        if ident not in seen:
            seen[ident] = None
    return list(seen.keys())


def _validate_characters(assertion: str) -> bool:
    """字符白名单校验：仅允许标识符/数字/比较逻辑运算符/括号/空白。

    Args:
        assertion: 断言字符串。

    Returns:
        全部字符在白名单内返回 True，否则 False。
    """
    return bool(_ALLOWED_CHAR_PATTERN.match(assertion))


def _validate_ast(tree: ast.AST) -> Tuple[bool, Optional[str]]:
    """AST 白名单校验：遍历所有节点，确保仅含安全节点类型。

    额外校验：
    - ast.Constant 的 value 仅允许 int/float/bool
    - ast.UnaryOp 的 op 仅允许 Not（拒绝 USub/UAdd 等算术一元运算）
    - ast.BoolOp 的 op 仅允许 And/Or
    - ast.Compare 的 ops 仅允许 Gt/Lt/GtE/LtE/Eq/NotEq

    Args:
        tree: ast.parse(assertion, mode='eval') 返回的 AST 根节点。

    Returns:
        (ok, bad_node_desc)：ok 为 True 表示通过；bad_node_desc 为违规节点描述。
    """
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            return False, type(node).__name__
        # Constant 值类型限制
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, bool)):
                return False, f"Constant({type(node.value).__name__})"
        # UnaryOp 操作符限制（仅 not）
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ast.Not):
                return False, f"UnaryOp({type(node.op).__name__})"
        # BoolOp 操作符限制（仅 and/or）
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, (ast.And, ast.Or)):
                return False, f"BoolOp({type(node.op).__name__})"
        # Compare 操作符限制（仅六种比较）
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if not isinstance(op, (ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq)):
                    return False, f"Compare({type(op).__name__})"
    return True, None


def _safe_eval_bool(assertion: str, simulation_metrics: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    """安全求值并返回布尔结果。

    执行顺序（任一步骤失败立即返回，不进入 eval）：
      1. 字符白名单校验（仅允许标识符/数字/比较逻辑运算符/括号/空白）
      2. ast.parse(mode='eval') 解析
      3. AST 节点白名单校验（Compare/BoolOp/UnaryOp/Name/Constant +
         And/Or/Not/Gt/Lt/GtE/LtE/Eq/NotEq）
      4. eval(assertion, {"__builtins__": {}}, simulation_metrics)
         ——仅在前三层通过后才执行，builtins 完全清空

    Args:
        assertion: 断言字符串。
        simulation_metrics: 仿真指标 dict（作为 locals 命名空间）。

    Returns:
        (result, status)：
        - result=True/False：求值成功，status="evaluated"
        - result=None：求值失败，status 为失败原因：
          ``"bad_chars"`` / ``"syntax_error"`` / ``"unsafe_node:<desc>"`` /
          ``"eval_error:<msg>"``
    """
    # 步骤 1：字符白名单
    if not _validate_characters(assertion):
        return None, "bad_chars"
    # 步骤 2：AST 解析
    try:
        tree = ast.parse(assertion, mode="eval")
    except SyntaxError:
        return None, "syntax_error"
    # 步骤 3：AST 节点白名单
    ok, bad_desc = _validate_ast(tree)
    if not ok:
        return None, f"unsafe_node:{bad_desc}"
    # 步骤 4：受限命名空间求值
    try:
        result = eval(assertion, {"__builtins__": {}}, simulation_metrics)  # noqa: S307
        return bool(result), "evaluated"
    except Exception as exc:  # noqa: BLE001
        return None, f"eval_error:{type(exc).__name__}:{exc}"


# =============================================================================
# 时间序列 → 峰值/活化时间提取
# =============================================================================
def _normalize_species_name(name: str) -> str:
    """物种名归一化：小写 + 将 - 与空格替换为 _。

    Args:
        name: 原始物种名（如 ``"pEGFR"`` / ``"Ras-GTP"`` / ``"RAS_GTP"``）。

    Returns:
        归一化后的物种名（如 ``"pegfr"`` / ``"ras_gtp"`` / ``"ras_gtp"``）。
    """
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _argmax_time(time: List[float], values: List[float]) -> Optional[float]:
    """取 values 峰值对应的 time（argmax 语义）。

    Args:
        time: 时间序列。
        values: 数值序列（与 time 等长）。

    Returns:
        峰值时刻（分钟，float）；空序列或长度不一致返回 None。
    """
    if not values or len(time) != len(values) or len(values) == 0:
        return None
    idx = max(range(len(values)), key=lambda i: values[i])
    return float(time[idx])


def _first_threshold_time(
    time: List[float], values: List[float], ratio: float = _ACTIVATION_THRESHOLD_RATIO
) -> Optional[float]:
    """取首个超过最大值 * ratio 的时刻（活化/表达启动语义）。

    用于 RAS_GTP（活化时间）与 DUSP（表达启动时间）等累积型响应：
    取首个达到峰值 50% 的时刻，比 argmax 更符合"活化时间"的生物学语义。

    Args:
        time: 时间序列。
        values: 数值序列（与 time 等长）。
        ratio: 阈值比例（默认 0.5）。

    Returns:
        首超阈值时刻（分钟，float）；空序列、全零序列或长度不一致返回 None。
    """
    if not values or len(time) != len(values) or len(values) == 0:
        return None
    max_v = max(values)
    if max_v <= 0:
        # 全零或全负：无有意义的活化时刻
        return None
    threshold = max_v * ratio
    for t, v in zip(time, values):
        if v >= threshold:
            return float(t)
    return None


def extract_peak_times_from_simulation(time_series: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
    """从仿真时间序列中提取各物种的峰值/活化时间。

    输入格式::

        {
            "pEGFR":  {"time": [0, 1, 2, ...], "values": [0.1, 0.5, 0.9, ...]},
            "ERK":    {"time": [...], "values": [...]},
            "RAS_GTP":{"time": [...], "values": [...]},
            "DUSP":   {"time": [...], "values": [...]},
        }

    输出格式::

        {
            "egfr_peak_time": 3.0,
            "erk_peak_time": 15.0,
            "ras_activation_time": 5.0,
            "dusp_expression_time": 30.0,
        }

    物种名归一化规则（大小写不敏感，- / 空格 视为 _）：
      - pEGFR / p_egfr / EGFR_p / EGFR → egfr_peak_time（argmax）
      - ERK / pERK / ppERK / p_erk / pp_erk → erk_peak_time（argmax）
      - RAS / RasGTP / RAS_GTP / Ras-GTP / RAS_GTP_active → ras_activation_time
        （首超最大值 50% 的时刻）
      - DUSP / DUSP1 → dusp_expression_time（首超最大值 50% 的时刻）

    若多个物种映射到同一指标键，后者覆盖前者（按 dict 迭代顺序）。

    Args:
        time_series: 仿真时间序列 dict。

    Returns:
        指标名 → 时间（分钟，float）的 dict。无法识别的物种名被忽略；
        空序列或长度不一致的条目返回 None（不写入结果 dict）。
    """
    result: Dict[str, float] = {}
    for species_name, series in time_series.items():
        if not isinstance(series, dict):
            logger.debug("跳过非 dict 时间序列条目: %s", species_name)
            continue
        time_list = series.get("time")
        values_list = series.get("values")
        if not isinstance(time_list, list) or not isinstance(values_list, list):
            logger.debug("跳过 time/values 非 list 的条目: %s", species_name)
            continue
        # 归一化物种名并查表
        norm = _normalize_species_name(species_name)
        metric_key = _SPECIES_TO_METRIC.get(norm)
        if metric_key is None:
            logger.debug("物种名 %r 未识别，跳过（归一化后: %r）", species_name, norm)
            continue
        # 根据 metric 语义选择提取策略
        if metric_key in _THRESHOLD_TIME_METRICS:
            t = _first_threshold_time(time_list, values_list)
        else:
            # 默认（含 _PEAK_TIME_METRICS）用 argmax
            t = _argmax_time(time_list, values_list)
        if t is not None:
            result[metric_key] = t
        else:
            logger.warning(
                "物种 %r 时间序列无效或全零，未能提取指标 %s",
                species_name, metric_key,
            )
    return result


# =============================================================================
# 主检查函数
# =============================================================================
def check_consistency(pathway: str, simulation_metrics: Dict[str, Any]) -> ConsistencyReport:
    """对仿真 metrics 做机制级一致性校验。

    流程：
      1. 从 Canonical 加载 pathway 的 consistency_rules
      2. 对每条规则：
         a. 提取 assertion 中的标识符，若任一不在 simulation_metrics 中 →
            标记 not_evaluated（不计违规，rules_evaluated 不递增）
         b. 字符白名单 + AST 白名单 + 受限 eval 求值
         c. 求值 True → 通过；求值 False → 记录 ConsistencyViolation
         d. 求值异常 → 记录带 "evaluation_error" label 的 violation（保守处理）
      3. passed = (无任何违规)，仅基于已评估规则

    Args:
        pathway: 通路标识（如 ``"egfr"``），传给 canonical_loader.get_consistency_rules。
        simulation_metrics: 仿真指标 dict，键为时间指标名（如
            ``"egfr_peak_time"``），值为分钟数（float）。assertion 中的变量名
            直接对应这些键。

    Returns:
        ConsistencyReport。若 Canonical 加载失败（文件不存在等），返回
        passed=True、rules_checked=0 的空报告（保守不阻塞主流程，仅记 warning）。
    """
    # 1. 加载 Canonical consistency_rules
    try:
        rules: List[ConsistencyRule] = get_consistency_rules(pathway)
    except CanonicalNotFoundError as exc:
        logger.warning(
            "Consistency Checker: pathway %r 的 Canonical 文件不存在，跳过检查: %s",
            pathway, exc,
        )
        return ConsistencyReport(
            pathway=pathway, passed=True, violations=[], rules_checked=0, rules_evaluated=0,
        )
    except Exception as exc:  # noqa: BLE001 —— Canonical 解析异常不应阻塞主流程
        logger.warning(
            "Consistency Checker: pathway %r Canonical 加载失败，跳过检查: %s",
            pathway, exc,
        )
        return ConsistencyReport(
            pathway=pathway, passed=True, violations=[], rules_checked=0, rules_evaluated=0,
        )

    report = ConsistencyReport(
        pathway=pathway,
        passed=True,
        violations=[],
        rules_checked=len(rules),
        rules_evaluated=0,
    )

    # 2. 逐条评估规则
    for rule in rules:
        assertion = rule.assertion.strip()
        if not assertion:
            logger.debug("规则 assertion 为空，跳过: %s", rule.rule)
            continue

        # 2a. 标识符缺失检查
        identifiers = _extract_identifiers(assertion)
        missing = [
            ident for ident in identifiers
            if ident not in _PY_KEYWORDS_AS_NAMES and ident not in simulation_metrics
        ]
        if missing:
            logger.debug(
                "规则 %r 因标识符缺失标记为 not_evaluated: %s",
                rule.rule, missing,
            )
            continue  # not_evaluated：不计违规，不递增 rules_evaluated

        # 2b/2c/2d. 安全求值
        result, status = _safe_eval_bool(assertion, simulation_metrics)
        if status == "evaluated":
            report.rules_evaluated += 1
            if result is True:
                # 规则通过
                continue
            # 2c. 求值 False → 记录违规
            observed: Dict[str, Any] = {
                ident: simulation_metrics[ident]
                for ident in identifiers
                if ident not in _PY_KEYWORDS_AS_NAMES
            }
            violation = ConsistencyViolation(
                rule=rule.rule,
                assertion=rule.assertion,
                violation_label=rule.violation_label or "consistency_violation",
                observed_values=observed,
                message=(
                    f"规则违规：{rule.rule}（断言 {rule.assertion!r} 求值为 False；"
                    f"观测值={observed}）"
                ),
            )
            report.violations.append(violation)
            logger.info(
                "Consistency Checker 发现违规 [%s]: %s",
                violation.violation_label, violation.message,
            )
        else:
            # 2d. 求值异常 → 保守记为 violation（label="evaluation_error"）
            observed_err: Dict[str, Any] = {
                ident: simulation_metrics[ident]
                for ident in identifiers
                if ident not in _PY_KEYWORDS_AS_NAMES
            }
            violation = ConsistencyViolation(
                rule=rule.rule,
                assertion=rule.assertion,
                violation_label="evaluation_error",
                observed_values=observed_err,
                message=(
                    f"规则求值异常：{rule.rule}（断言 {rule.assertion!r}，"
                    f"失败原因={status}，观测值={observed_err}）"
                ),
            )
            report.violations.append(violation)
            logger.warning(
                "Consistency Checker 规则求值异常 [%s]: %s",
                status, violation.message,
            )

    # 3. passed 判定：已评估规则中无任何违规
    report.passed = (len(report.violations) == 0)
    return report


__all__ = [
    "ConsistencyViolation",
    "ConsistencyReport",
    "check_consistency",
    "extract_peak_times_from_simulation",
]
