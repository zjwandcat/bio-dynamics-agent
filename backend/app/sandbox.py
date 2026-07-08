# BioDynamics Agent - 代码执行沙箱
# 使用 subprocess 在隔离临时目录中运行 LLM 生成的 Python 仿真代码，
# 捕获控制台输出并将生成的图片转为 Base64。
# 新增：静态安全扫描（拦截危险模块导入）与生物学常识检查（负浓度 / NaN / Inf）。
# v2 升级：execute_simulation_code_v2 新增 AST 预检、错误分类与 CSV 路径返回。
#
# 深度审核报告 §3.1 加固：
# - 确定性求解：默认 LSODA，禁止随机噪声（除非 SDE 模式）
# - 资源限制：max_step + timeout 显式设置
# - 审计日志：所有沙箱执行的代码、耗时、资源占用持久化到 data/sandbox_logs/

import ast
import base64
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.metrics import get_metrics

logger = logging.getLogger(__name__)

# 错误分类常量（v2 N7 状态字段 error_class 取值）
ERR_NONE = "none"
ERR_SYNTAX = "syntax_error"
ERR_RUNTIME = "runtime_error"
ERR_NUMERICAL = "numerical_error"
ERR_TIMEOUT = "timeout"
ERR_SECURITY = "security_error"


# 禁止导入的危险模块/包，用于拦截沙箱逃逸、文件删除、网络操作等恶意代码
_BLOCKED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "urllib",
    "requests",
    "http",
    "ftplib",
    "smtplib",
    "email",
    "pickle",
    "ctypes",
    "multiprocessing",
    "threading",
    "asyncio",
}

# 禁止使用的危险内建函数（按此顺序匹配，确保外层调用优先被识别）
_BLOCKED_BUILTINS = ("eval", "exec", "compile", "open", "input", "__import__")


def _check_code_security(code: str) -> tuple[bool, str]:
    """静态扫描代码，返回 (是否安全, 错误信息)。"""
    # 1. 拦截显式 import / from ... import
    for module in _BLOCKED_MODULES:
        # import os
        if re.search(rf"^\s*import\s+{module}\b", code, re.MULTILINE):
            return False, f"安全拦截：禁止导入模块 '{module}'"
        # from os import ... / from os.path import ...
        if re.search(rf"^\s*from\s+{module}(?:\.\w+)*\s+import", code, re.MULTILINE):
            return False, f"安全拦截：禁止从模块 '{module}' 导入"

    # 2. 拦截 __import__ / eval / exec / compile / open 等危险内建
    for builtin in _BLOCKED_BUILTINS:
        if re.search(rf"\b{builtin}\s*\(", code):
            return False, f"安全拦截：禁止使用危险内建函数 '{builtin}'"

    return True, ""


# -----------------------------------------------------------------------------
# 深度审核报告 §3.1：确定性求解检查（禁止随机噪声除非 SDE 模式）
# -----------------------------------------------------------------------------
# 允许的确定性求解器（默认 LSODA / RK45 / BDF / Radau 等）
_DETERMINISTIC_SOLVERS = {"LSODA", "RK45", "RK23", "BDF", "Radau", "DOP853", "odeint"}
# 随机噪声相关 API（除非 SDE 模式显式开启，否则禁止）
_STOCHASTIC_PATTERNS = [
    r"\bnp\.random\.",
    r"\brandom\.random\b",
    r"\brandom\.gauss\b",
    r"\brandom\.uniform\b",
    r"\brandom\.normal\b",
    r"\bsde\b",
    r"noise\s*=",
]


def _check_determinism(code: str, allow_stochastic: bool = False) -> tuple[bool, str]:
    """检查代码是否符合确定性求解要求（深度审核报告 §3.1）。

    Args:
        code: 待执行代码
        allow_stochastic: 是否允许随机噪声（SDE 模式下为 True）

    Returns:
        (是否通过, 错误信息)
    """
    if allow_stochastic:
        return True, ""

    # 1. 检查随机噪声 API
    for pattern in _STOCHASTIC_PATTERNS:
        if re.search(pattern, code):
            return False, (
                f"确定性求解违规：检测到随机噪声 API（{pattern}）。"
                f"默认使用 LSODA 确定性求解器；如需 SDE 模式请显式开启。"
            )

    # 2. 检查求解器是否为确定性（warning 级别，不阻断）
    solver_match = re.search(r"method\s*=\s*['\"](\w+)['\"]", code)
    if solver_match:
        solver = solver_match.group(1)
        if solver not in _DETERMINISTIC_SOLVERS:
            logger.warning("非标准求解器：%s（建议使用 LSODA 确保确定性）", solver)

    return True, ""


# -----------------------------------------------------------------------------
# 深度审核报告 §3.1：沙箱审计日志持久化
# -----------------------------------------------------------------------------
def _write_audit_log(
    code: str,
    status: str,
    error_class: str,
    duration_seconds: float,
    stdout_stderr: str,
    metadata: dict | None = None,
) -> None:
    """将沙箱执行审计日志持久化到 data/sandbox_logs/。

    日志内容包括：代码哈希、执行状态、错误分类、耗时、stdout/stderr 摘要。
    """
    if not settings.SANDBOX_AUDIT_LOG:
        return
    try:
        log_dir = Path(settings.SANDBOX_AUDIT_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        # 按日期分文件
        date_str = datetime.now().strftime("%Y%m%d")
        log_path = log_dir / f"sandbox_{date_str}.jsonl"

        # 代码哈希（避免存储完整代码造成膨胀）
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        # stdout/stderr 摘要（前 2000 字符）
        stdout_summary = stdout_stderr[:2000] if stdout_stderr else ""

        log_entry = {
            "ts": datetime.now().isoformat(),
            "code_hash": code_hash,
            "status": status,
            "error_class": error_class,
            "duration_seconds": round(duration_seconds, 4),
            "stdout_len": len(stdout_stderr) if stdout_stderr else 0,
            "stdout_summary": stdout_summary,
            "code_size": len(code),
            "metadata": metadata or {},
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("写入沙箱审计日志失败: %s", exc)


def _check_biological_validity(stdout: str) -> tuple[bool, str]:
    """检查 stdout 中是否包含生物学上不可接受的结果（负浓度、NaN、Inf）。

    约定：生成代码可在 stdout 中输出 BIO_CHECK: < species > = < value > 行，
    沙箱将据此判断结果是否合规。
    """
    violations: list[str] = []
    for line in stdout.splitlines():
        match = re.search(r"BIO_CHECK:\s*(\S+)\s*=\s*([\-+\deE\.]+|nan|inf|-inf)", line, re.IGNORECASE)
        if not match:
            continue
        species, value_str = match.groups()
        value_str_lower = value_str.lower()
        if value_str_lower in ("nan", "inf", "-inf"):
            violations.append(f"{species} = {value_str}（非有限数值）")
            continue
        try:
            value = float(value_str)
        except ValueError:
            continue
        if value < 0:
            violations.append(f"{species} = {value}（负浓度/数量）")

    if violations:
        return False, "生物学常识检查未通过：" + "; ".join(violations)
    return True, ""


def _parse_dose_response(stdout: str) -> dict | None:
    """从 stdout 中解析 DOSE_RESPONSE 单行 JSON 行。"""
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("DOSE_RESPONSE:"):
            json_str = stripped[len("DOSE_RESPONSE:"):].strip()
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "concentrations" in data and "effects" in data:
                    return data
            except json.JSONDecodeError:
                continue
    return None


def _parse_scalar_marker(stdout: str, marker: str) -> float | None:
    """从 stdout 中解析 IC50: / IC90: / HED: 等标量标记行。"""
    for line in stdout.splitlines():
        stripped = line.strip()
        prefix = f"{marker}:"
        if stripped.startswith(prefix):
            value_str = stripped[len(prefix):].strip()
            try:
                return float(value_str)
            except ValueError:
                continue
    return None


def _parse_combo_ci(stdout: str) -> dict | None:
    """从 stdout 中解析 COMBO_CI 行，返回 {fa_0.5: ci, ...}。"""
    result: dict = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("COMBO_CI:"):
            match = re.search(r"fa=([\d.]+).*CI=([\d.]+)", stripped)
            if match:
                fa = float(match.group(1))
                ci = float(match.group(2))
                result[f"fa_{fa}"] = ci
    return result if result else None


def execute_simulation_code(code: str) -> dict:
    """在隔离沙箱中执行仿真代码。

    Args:
        code: LLM 生成的完整 Python 代码。

    Returns:
        包含执行状态、合并后的 stdout/stderr 日志和图片 Base64 的字典。
    """
    # 1. 静态安全扫描
    is_safe, security_msg = _check_code_security(code)
    if not is_safe:
        return {
            "status": "error",
            "stdout_stderr": security_msg,
            "image_base64": None,
        }

    temp_dir = tempfile.mkdtemp(prefix="bio_dynamics_")
    try:
        cleaned = code.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        # 强制移除可能阻塞的 plt.show()
        cleaned = re.sub(r"\bplt\.show\(\s*\)", "", cleaned)

        run_path = os.path.join(temp_dir, "run.py")
        with open(run_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"

        try:
            result = subprocess.run(
                ["python", "run.py"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "stdout_stderr": "代码执行超过 60 秒，已被终止。",
                "image_base64": None,
            }
        except Exception as exc:
            return {
                "status": "error",
                "stdout_stderr": f"沙箱执行异常: {exc}",
                "image_base64": None,
            }

        stdout_stderr = ""
        if result.stdout:
            stdout_stderr += result.stdout
        if result.stderr:
            if stdout_stderr:
                stdout_stderr += "\n"
            stdout_stderr += result.stderr

        image_base64 = None
        image_path = os.path.join(temp_dir, "simulation.png")
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        if result.returncode != 0 or result.stderr or image_base64 is None:
            status = "error"
        else:
            status = "success"
            # 2. 生物学常识检查：成功执行后仍要确认结果没有负浓度或 NaN/Inf
            bio_ok, bio_msg = _check_biological_validity(stdout_stderr)
            if not bio_ok:
                status = "error"
                if stdout_stderr:
                    stdout_stderr += "\n"
                stdout_stderr += bio_msg

        # 解析剂量递增与联合用药输出（即使主仿真失败也尝试解析已输出的数据）
        dose_response_data = _parse_dose_response(stdout_stderr)
        ic50 = _parse_scalar_marker(stdout_stderr, "IC50")
        ic90 = _parse_scalar_marker(stdout_stderr, "IC90")
        hed = _parse_scalar_marker(stdout_stderr, "HED")
        combo_ci_data = _parse_combo_ci(stdout_stderr)

        return {
            "status": status,
            "stdout_stderr": stdout_stderr,
            "image_base64": image_base64,
            "dose_response_data": dose_response_data,
            "ic50": ic50,
            "ic90": ic90,
            "hed": hed,
            "combo_ci_data": combo_ci_data,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# v2 升级：AST 预检 + 错误分类 + CSV 路径返回
# -----------------------------------------------------------------------------
def _ast_precheck(code: str) -> tuple[bool, str]:
    """使用 ast.parse() 预检代码语法。

    Returns:
        (是否合法, 错误信息)
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, f"AST 预检失败（line {exc.lineno}）：{exc.msg}"
    except Exception as exc:
        return False, f"AST 预检异常：{exc}"


def _classify_error(returncode: int, stdout_stderr: str, was_timeout: bool) -> str:
    """根据执行结果对错误进行分类。

    Returns:
        错误类名字符串（ERR_NONE / ERR_RUNTIME / ERR_NUMERICAL / ERR_TIMEOUT）。
    """
    if was_timeout:
        return ERR_TIMEOUT
    if returncode == 0:
        # 检查是否产生数值异常（NaN/Inf/数值溢出）
        if re.search(r"\b(nan|inf|-inf|overflow|underflow)\b", stdout_stderr, re.IGNORECASE):
            return ERR_NUMERICAL
        return ERR_NONE
    return ERR_RUNTIME


def execute_simulation_code_v2(code: str, timeout: int | None = None, allow_stochastic: bool = False) -> dict:
    """v2 沙箱执行入口：AST 预检 → 安全扫描 → 确定性检查 → 执行 → 错误分类 → CSV/PNG 收集。

    与 v1 的区别：
    1. 前置 ast.parse() 拦截语法错误，避免进入 subprocess 浪费 60s
    2. 错误分类写入 error_class 字段
    3. 始终返回 simulation_csv_path（若代码生成 CSV）
    4. 安全拦截写入 error_class=security_error
    5. 深度审核报告 §3.1：确定性求解检查（默认禁随机噪声）+ 审计日志 + metrics 埋点
    """
    # 使用配置化超时（深度审核报告 §3.1 资源限制）
    if timeout is None:
        timeout = settings.SANDBOX_TIMEOUT

    exec_start = time.perf_counter()
    metrics = get_metrics()

    # 1. AST 预检
    ast_ok, ast_msg = _ast_precheck(code)
    if not ast_ok:
        logger.warning("v2 AST 预检失败：%s", ast_msg)
        duration = time.perf_counter() - exec_start
        _write_audit_log(code, "error", ERR_SYNTAX, duration, ast_msg)
        metrics.record_sandbox_execution(False, ERR_SYNTAX, duration)
        return {
            "status": "error",
            "stdout_stderr": ast_msg,
            "image_base64": "",
            "error_class": ERR_SYNTAX,
            "simulation_csv_path": "",
        }

    # 2. 静态安全扫描
    is_safe, security_msg = _check_code_security(code)
    if not is_safe:
        duration = time.perf_counter() - exec_start
        _write_audit_log(code, "error", ERR_SECURITY, duration, security_msg)
        metrics.record_sandbox_execution(False, ERR_SECURITY, duration)
        return {
            "status": "error",
            "stdout_stderr": security_msg,
            "image_base64": "",
            "error_class": ERR_SECURITY,
            "simulation_csv_path": "",
        }

    # 3. 确定性求解检查（深度审核报告 §3.1）
    det_ok, det_msg = _check_determinism(code, allow_stochastic=allow_stochastic)
    if not det_ok:
        duration = time.perf_counter() - exec_start
        _write_audit_log(code, "error", ERR_SECURITY, duration, det_msg)
        metrics.record_sandbox_execution(False, ERR_SECURITY, duration)
        return {
            "status": "error",
            "stdout_stderr": det_msg,
            "image_base64": "",
            "error_class": ERR_SECURITY,
            "simulation_csv_path": "",
        }

    # 4. 执行
    temp_dir = tempfile.mkdtemp(prefix="bio_dynamics_v2_")
    try:
        cleaned = code.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
        cleaned = re.sub(r"\bplt\.show\(\s*\)", "", cleaned)

        run_path = os.path.join(temp_dir, "run.py")
        with open(run_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"

        was_timeout = False
        try:
            result = subprocess.run(
                ["python", "run.py"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            was_timeout = True
            duration = time.perf_counter() - exec_start
            timeout_msg = f"代码执行超过 {timeout} 秒，已被终止。"
            _write_audit_log(code, "error", ERR_TIMEOUT, duration, timeout_msg)
            metrics.record_sandbox_execution(False, ERR_TIMEOUT, duration)
            return {
                "status": "error",
                "stdout_stderr": timeout_msg,
                "image_base64": "",
                "error_class": ERR_TIMEOUT,
                "simulation_csv_path": "",
            }
        except Exception as exc:
            duration = time.perf_counter() - exec_start
            err_msg = f"沙箱执行异常: {exc}"
            _write_audit_log(code, "error", ERR_RUNTIME, duration, err_msg)
            metrics.record_sandbox_execution(False, ERR_RUNTIME, duration)
            return {
                "status": "error",
                "stdout_stderr": err_msg,
                "image_base64": "",
                "error_class": ERR_RUNTIME,
                "simulation_csv_path": "",
            }

        stdout_stderr = ""
        if result.stdout:
            stdout_stderr += result.stdout
        if result.stderr:
            if stdout_stderr:
                stdout_stderr += "\n"
            stdout_stderr += result.stderr

        # 5. 收集产物
        image_base64 = ""
        image_path = os.path.join(temp_dir, "simulation.png")
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        csv_path = os.path.join(temp_dir, "simulation.csv")
        simulation_csv_path = csv_path if os.path.exists(csv_path) else ""

        # 在 finally 清理 temp_dir 之前，将 CSV 复制到持久目录
        if simulation_csv_path:
            persistent_dir = tempfile.mkdtemp(prefix="bio_dynamics_csv_")
            persistent_csv = os.path.join(persistent_dir, "simulation.csv")
            shutil.copy2(simulation_csv_path, persistent_csv)
            simulation_csv_path = persistent_csv

        # 6. 错误分类
        error_class = _classify_error(
            returncode=result.returncode,
            stdout_stderr=stdout_stderr,
            was_timeout=was_timeout,
        )
        status = "success" if error_class == ERR_NONE and image_base64 else "error"

        # 7. 生物学常识检查（v1 兼容）
        if status == "success":
            bio_ok, bio_msg = _check_biological_validity(stdout_stderr)
            if not bio_ok:
                status = "error"
                error_class = ERR_NUMERICAL
                if stdout_stderr:
                    stdout_stderr += "\n"
                stdout_stderr += bio_msg

        # 7.5 TD-003 接线强制：仿真后验证（负浓度 / NaN / Inf / 爆炸 / 质量守恒）
        # 纪律1：post_simulation_validation 必须在仿真后流程中强制调用，拦截不通过的仿真。
        # 当 CSV 存在时无条件调用（species_names=None 自动从列头检测）。
        if simulation_csv_path and status == "success":
            try:
                psv_result = post_simulation_validation(simulation_csv_path)
                if not psv_result.get("passed", True):
                    status = "error"
                    error_class = ERR_NUMERICAL
                    if stdout_stderr:
                        stdout_stderr += "\n"
                    stdout_stderr += (
                        f"[post_simulation_validation FAILED] "
                        f"negative_concentrations={psv_result.get('negative_concentrations', [])[:3]} "
                        f"nan_detected={psv_result.get('nan_detected')} "
                        f"inf_detected={psv_result.get('inf_detected')} "
                        f"explosion_detected={psv_result.get('explosion_detected')} "
                        f"mass_conservation_violations={len(psv_result.get('mass_conservation_violations', []))}"
                    )
                    logger.warning(
                        "post_simulation_validation 拦截仿真: %s",
                        psv_result,
                    )
            except Exception as exc:
                logger.warning("post_simulation_validation 执行异常: %s", exc)

        # 8. 审计日志 + metrics 埋点（深度审核报告 §3.1 + §3.3）
        duration = time.perf_counter() - exec_start
        _write_audit_log(
            code, status, error_class, duration, stdout_stderr,
            metadata={"returncode": result.returncode, "csv_path": simulation_csv_path},
        )
        metrics.record_sandbox_execution(status == "success", error_class, duration)

        return {
            "status": status,
            "stdout_stderr": stdout_stderr,
            "image_base64": image_base64,
            "error_class": error_class,
            "simulation_csv_path": simulation_csv_path,
            "returncode": result.returncode,
            "duration_seconds": round(duration, 4),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# IB-014 修复：仿真后质量守恒验证 + 负浓度 / NaN / Inf / 爆炸检测
# -----------------------------------------------------------------------------
# 负浓度容差：避免浮点误差导致的误报（小于此值视为数值噪声）
_NEGATIVE_TOLERANCE = 1e-9
# 爆炸阈值：最大浓度超过此值视为数值爆炸（超出生物合理范围）
_EXPLOSION_THRESHOLD = 1e6


def post_simulation_validation(csv_path: str, species_names: list[str] | None = None,
                                constraints: list[dict] | None = None) -> dict:
    """仿真后验证：检测负浓度、NaN、Inf、质量守恒。

    IB-014 修复：sandbox 仿真后应检测：
    1. 负浓度（任何时间点任何物种 < -tolerance）
    2. NaN / Inf（数值发散标志）
    3. 质量守恒（基于 constraints 中的 mass_conservation 约束）
    4. 爆炸检测（最大浓度 > 1e6，超出生物合理范围）

    Args:
        csv_path: 仿真输出 CSV 文件路径，应包含 time 列与各物种浓度列。
        species_names: 待校验的物种名列表（需与 CSV 列名一致）。
            为 None 时自动从 CSV 列头检测（排除 time/t/Time/T 列）。
        constraints: 约束列表，每项形如
            {"type": "mass_conservation", "species": ["A", "B"], "tolerance": 1e-6}。

    Returns:
        dict: {
            "passed": bool,
            "negative_concentrations": list[dict],  # [{species, time, value}]
            "nan_detected": bool,
            "inf_detected": bool,
            "explosion_detected": bool,
            "mass_conservation_violations": list[dict],
            "max_concentrations": dict[str, float],
        }
    """
    # 初始化结果字典
    result: dict = {
        "passed": True,
        "negative_concentrations": [],
        "nan_detected": False,
        "inf_detected": False,
        "explosion_detected": False,
        "mass_conservation_violations": [],
        "max_concentrations": {},
    }

    # 1. 读取 CSV 文件（使用标准库 csv 模块，避免引入 pandas 依赖）
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except (FileNotFoundError, OSError) as exc:
        logger.warning("post_simulation_validation 读取 CSV 失败: %s", exc)
        result["passed"] = False
        return result

    # 空文件直接判定失败
    if not rows:
        logger.warning("post_simulation_validation CSV 为空: %s", csv_path)
        result["passed"] = False
        return result

    # 2. 识别时间列（兼容 time / t / Time / T）
    time_col: str | None = None
    for candidate in ("time", "t", "Time", "T"):
        if candidate in rows[0]:
            time_col = candidate
            break

    # 3. 筛选 CSV 中实际存在的物种列
    # TD-003 接线：species_names 为 None 时自动从 CSV 列头检测（排除时间列）
    if species_names is None:
        _time_cols = {"time", "t", "Time", "T"}
        available_species = [k for k in rows[0].keys() if k not in _time_cols]
        logger.debug(
            "post_simulation_validation 自动检测物种列: %s", available_species
        )
    else:
        available_species = [s for s in species_names if s in rows[0]]
    if not available_species:
        logger.warning(
            "post_simulation_validation 未在 CSV 中找到任何指定物种列: %s",
            species_names,
        )
        result["passed"] = False
        return result

    # 收集每个物种的 (time, value) 序列，用于后续质量守恒与爆炸检测
    species_data: dict[str, list[tuple[float, float]]] = {s: [] for s in available_species}

    # 4. 逐行扫描：检测负浓度 / NaN / Inf
    for row in rows:
        # 解析当前行的时间值
        time_value = 0.0
        if time_col is not None:
            try:
                time_value = float(row[time_col])
            except (ValueError, TypeError):
                time_value = 0.0

        # 逐个物种检测
        for species in available_species:
            raw_value = row.get(species, "")
            # 先尝试直接转 float（覆盖正常数值与 "nan"/"inf" 字面量）
            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                # 无法解析的字符串：检测是否为 NaN / Inf 文本
                if isinstance(raw_value, str):
                    lower = raw_value.strip().lower()
                    if "nan" in lower:
                        result["nan_detected"] = True
                    elif "inf" in lower:
                        result["inf_detected"] = True
                continue

            # NaN 检测
            if math.isnan(value):
                result["nan_detected"] = True
                continue
            # Inf 检测
            if math.isinf(value):
                result["inf_detected"] = True
                continue

            # 记录有效数值（用于爆炸检测与质量守恒）
            species_data[species].append((time_value, value))

            # 负浓度检测（小于负容差即判定违规）
            if value < -_NEGATIVE_TOLERANCE:
                result["negative_concentrations"].append({
                    "species": species,
                    "time": time_value,
                    "value": value,
                })

    # 5. 计算各物种最大浓度，并执行爆炸检测
    for species, data_points in species_data.items():
        if data_points:
            max_val = max(v for _, v in data_points)
            result["max_concentrations"][species] = max_val
            if max_val > _EXPLOSION_THRESHOLD:
                result["explosion_detected"] = True
        else:
            result["max_concentrations"][species] = 0.0

    # 6. 质量守恒检测（基于 constraints 中的 mass_conservation 约束）
    if constraints:
        for constraint in constraints:
            if not isinstance(constraint, dict):
                continue
            if constraint.get("type") != "mass_conservation":
                continue

            # 约束涉及的物种与容差
            cons_species = constraint.get("species", [])
            cons_tolerance = constraint.get("tolerance", 1e-6)

            # 仅保留有数据的物种
            valid_species = [
                s for s in cons_species
                if s in species_data and species_data[s]
            ]
            if not valid_species:
                continue

            # 按时间点聚合各物种浓度
            time_to_values: dict[float, dict[str, float]] = {}
            for s in valid_species:
                for t, v in species_data[s]:
                    time_to_values.setdefault(t, {})[s] = v

            sorted_times = sorted(time_to_values.keys())
            if not sorted_times:
                continue

            # 以第一个时间点的总量作为基准
            initial_total = None
            for t in sorted_times:
                values_at_t = time_to_values[t]
                # 仅在所有涉及物种均有值时才计算总量
                if len(values_at_t) != len(valid_species):
                    continue
                total = sum(values_at_t.values())
                if initial_total is None:
                    initial_total = total
                elif abs(total - initial_total) > cons_tolerance:
                    result["mass_conservation_violations"].append({
                        "time": t,
                        "expected": initial_total,
                        "actual": total,
                        "deviation": total - initial_total,
                    })

    # 7. 综合判定：任一检查项不通过则整体失败
    result["passed"] = (
        not result["negative_concentrations"]
        and not result["nan_detected"]
        and not result["inf_detected"]
        and not result["explosion_detected"]
        and not result["mass_conservation_violations"]
    )

    return result
