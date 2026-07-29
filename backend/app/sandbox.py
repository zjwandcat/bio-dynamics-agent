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
import sys
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
ERR_RECURSION = "recursion_error"  # [P0-3] 代码逻辑 bug，不可重试
ERR_PERMISSION = "permission_error"  # [Sandbox Fix] Windows 权限拒绝 / ACL 拦截
ERR_IMPORT_BLOCKED = "import_blocked"  # [Sandbox N1] import guard 拦截非白名单模块


# -----------------------------------------------------------------------------
# [Sandbox N1] subprocess 隔离增强：资源限制默认值
# -----------------------------------------------------------------------------
# 可通过环境变量 SANDBOX_MEM_MB / SANDBOX_CPU_SEC / SANDBOX_FILE_MB 覆盖
# 失败时（平台不支持）记录 warning，不阻断执行
_SANDBOX_DEFAULT_MEM_MB = int(os.getenv("SANDBOX_MEM_MB", "2048"))     # 2GB
_SANDBOX_DEFAULT_CPU_SEC = int(os.getenv("SANDBOX_CPU_SEC", "120"))    # 120 秒
_SANDBOX_DEFAULT_FILE_MB = int(os.getenv("SANDBOX_FILE_MB", "100"))    # 100MB
_SANDBOX_DEFAULT_NPROC = 1                                             # 单进程

# import guard 拦截签名（与 sandbox_safe_imports._BLOCKED_SIGNAL 保持一致）
_IMPORT_BLOCKED_SIGNATURE = "[ImportBlocked]"


def _build_preexec_fn(
    mem_mb: int = _SANDBOX_DEFAULT_MEM_MB,
    cpu_sec: int = _SANDBOX_DEFAULT_CPU_SEC,
    file_mb: int = _SANDBOX_DEFAULT_FILE_MB,
    nproc: int = _SANDBOX_DEFAULT_NPROC,
):
    """构建子进程 preexec_fn，在 Unix 平台设置 RLIMIT_AS / RLIMIT_CPU / RLIMIT_FSIZE / RLIMIT_NPROC。

    Args:
        mem_mb: 内存上限（MB），0 表示不限制
        cpu_sec: CPU 时间上限（秒），0 表示不限制
        file_mb: 单文件大小上限（MB），0 表示不限制
        nproc: 进程数上限，0 表示不限制

    Returns:
        preexec_fn 可调用对象，或 None（Windows / resource 不可用时）。

    Note:
        Windows 不支持 preexec_fn（subprocess.run 会抛 ValueError）。
        Windows 下的资源限制需要通过 Job Object 实现（psutil 可选），
        本函数返回 None，由调用方记录 warning 降级运行。
    """
    if sys.platform == "win32":
        return None
    try:
        import resource  # Unix-only
    except ImportError:
        logger.warning(
            "[Sandbox N1] resource 模块不可用（platform=%s），资源限制降级",
            sys.platform,
        )
        return None

    mem_bytes = mem_mb * 1024 * 1024 if mem_mb > 0 else 0
    file_bytes = file_mb * 1024 * 1024 if file_mb > 0 else 0

    def _set_limits():
        try:
            if mem_bytes > 0:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            if cpu_sec > 0:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
            if file_bytes > 0:
                resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
            if nproc > 0:
                resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        except (ValueError, OSError) as exc:
            # 子进程无法访问父进程 logger，写入 stderr 供审计
            import sys as _sys
            _sys.stderr.write(f"[Sandbox N1] resource setrlimit warning: {exc}\n")

    return _set_limits


def _run_subprocess_with_escalation(
    cmd: list[str],
    cwd: str,
    env: dict,
    timeout: int,
    preexec_fn=None,
) -> tuple[int, str, str, bool]:
    """使用 Popen 执行子进程，超时后 SIGTERM → SIGKILL 升级 kill（Windows 用 taskkill /T）。

    与 subprocess.run 的区别：
        1. 超时后递归 kill 子进程树（subprocess.run 只 kill 直接子进程）
        2. Windows 下用 taskkill /F /T /PID 杀整个进程树
        3. 返回 (returncode, stdout, stderr, was_timeout) 元组，统一成功/超时接口

    Args:
        cmd: 命令列表，如 ["python", "run.py"]
        cwd: 工作目录
        env: 环境变量字典
        timeout: 超时秒数
        preexec_fn: 子进程启动前回调（Unix 资源限制）

    Returns:
        (returncode, stdout_text, stderr_text, was_timeout)
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        preexec_fn=preexec_fn if sys.platform != "win32" else None,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout or "", stderr or "", False
    except subprocess.TimeoutExpired:
        # 升级 kill：先尝试 graceful（SIGTERM / TerminateProcess），失败则 force kill
        was_timeout = True
        stdout_text = ""
        stderr_text = ""
        try:
            if sys.platform == "win32":
                # Windows: taskkill /F /T /PID 递归 kill 整个进程树
                # /F = force, /T = tree (kill child processes recursively)
                kill_result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if kill_result.returncode != 0:
                    # taskkill 失败 → fallback 到 proc.kill()
                    proc.kill()
            else:
                # Unix: SIGTERM → wait 2s → SIGKILL
                proc.terminate()
                try:
                    stdout_text, stderr_text = proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception as exc:
            # 兜底：force kill，避免僵尸进程
            logger.warning("[Sandbox N1] 子进程 kill 异常: %s", exc)
            try:
                proc.kill()
            except Exception:
                pass

        # 最终回收（确保子进程已退出，避免僵尸）
        try:
            stdout_text, stderr_text = proc.communicate(timeout=5)
        except Exception:
            pass

        return (
            proc.returncode if proc.returncode is not None else -1,
            stdout_text or "",
            stderr_text or "",
            was_timeout,
        )


def _detect_import_blocked(stderr: str) -> bool:
    """检测 stderr 是否包含 import guard 拦截签名 [ImportBlocked]。

    sandbox_safe_imports.py 拦截非白名单 import 时，会在 ImportError 消息中
    写入 [ImportBlocked] 签名。本函数据此判断是否为 import 拦截失败。

    Args:
        stderr: 子进程 stderr 文本

    Returns:
        True 如果检测到 [ImportBlocked] 签名
    """
    if not stderr:
        return False
    return _IMPORT_BLOCKED_SIGNATURE in stderr


def _build_import_guard_bootstrap(work_dir: str) -> str:
    """构建 run.py 启动头，注入 import guard 安装代码。

    将 sandbox_safe_imports.py 复制到 work_dir，然后在 run.py 头部注入：
        1. 从 sandbox_safe_imports 导入 install_import_guard
        2. 调用 install_import_guard() 安装 import 拦截器
        3. 清理临时符号（避免污染用户命名空间）

    Args:
        work_dir: 沙箱工作目录（sandbox_safe_imports.py 复制目标）

    Returns:
        注入到 run.py 头部的 bootstrap 代码字符串
    """
    # 复制 sandbox_safe_imports.py 到 work_dir
    # （避免将 backend/app/ 加入 sys.path 暴露 config.py / sandbox.py 等敏感模块）
    safe_imports_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sandbox_safe_imports.py"
    )
    safe_imports_dst = os.path.join(work_dir, "sandbox_safe_imports.py")
    try:
        shutil.copy2(safe_imports_src, safe_imports_dst)
    except (OSError, PermissionError) as exc:
        logger.warning(
            "[Sandbox N1] sandbox_safe_imports.py 复制失败 (%s): %s",
            safe_imports_dst, exc,
        )
        # 复制失败时仍尝试从原路径加载（backend_root 已在 PYTHONPATH，但 app/ 未在）
        # 这种情况下 guard 安装可能失败，子进程会以无 guard 模式运行（降级）
        return ""

    # run.py 运行时 cwd=work_dir，Python 自动将 script dir (work_dir) 加入 sys.path[0]
    # 因此 from sandbox_safe_imports import ... 无需显式 sys.path 操作
    bootstrap = (
        "# [Sandbox N1] import guard bootstrap (auto-injected)\n"
        "try:\n"
        "    from sandbox_safe_imports import install_import_guard as _install_guard\n"
        "    _install_guard()\n"
        "    del _install_guard\n"
        "except Exception as _guard_err:\n"
        "    import sys as _sys\n"
        "    _sys.stderr.write(f'[Sandbox N1] import guard install failed: {_guard_err}\\n')\n"
        "    del _sys, _guard_err\n"
        "# --- end bootstrap ---\n"
    )
    return bootstrap


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
    """静态扫描代码，返回 (是否安全, 错误信息)。

    Security decisions are made from the syntax tree so words inside comments,
    docstrings, and ordinary string literals cannot reject otherwise valid ODE
    programs.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"安全扫描语法错误（line {exc.lineno}）：{exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _BLOCKED_MODULES:
                    return False, f"安全拦截：禁止导入模块 '{root}'"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in _BLOCKED_MODULES:
                return False, f"安全拦截：禁止从模块 '{root}' 导入"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_BUILTINS:
                return False, f"安全拦截：禁止使用危险内建函数 '{node.func.id}'"

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
        # [P0-2 修复] stdout/stderr 摘要：保留前 1000 + 后 1000 字符
        # 旧实现仅保留前 2000 字符，导致 Traceback 的异常类型行被截断丢失。
        # 新实现：错误时保留尾部（含异常类型与消息），成功时保留头部。
        if not stdout_stderr:
            stdout_summary = ""
        elif len(stdout_stderr) <= 2000:
            stdout_summary = stdout_stderr
        else:
            head = stdout_stderr[:1000]
            tail = stdout_stderr[-1000:]
            stdout_summary = f"{head}\n...[truncated {len(stdout_stderr) - 2000} chars]...\n{tail}"

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
        # Generated models contain biological Unicode identifiers (NF-κB,
        # IκBα, ∅).  Force UTF-8 mode so redirected stdout and numpy.savetxt
        # do not fall back to the Windows GBK locale and leave a zero-byte CSV.
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # [BM2-BM8 修复 / Mode B] v4_ode_system 可能 import app.solvers.dde_solver 等
        # 模块，sandbox 临时目录不包含 app 包。将 backend 根目录加入 PYTHONPATH，
        # 使 sandbox 内可 import app.* 模块（仅 v4 ODE 模板需要，v3 模板不受影响）。
        backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = backend_root + os.pathsep + env["PYTHONPATH"]
        else:
            env["PYTHONPATH"] = backend_root

        try:
            result = subprocess.run(
                [sys.executable, "run.py"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
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
        错误类名字符串（ERR_NONE / ERR_RUNTIME / ERR_NUMERICAL / ERR_TIMEOUT / ERR_RECURSION）。
    """
    if was_timeout:
        return ERR_TIMEOUT
    if returncode == 0:
        # 检查是否产生数值异常（NaN/Inf/数值溢出）
        if re.search(r"\b(nan|inf|-inf|overflow|underflow)\b", stdout_stderr, re.IGNORECASE):
            return ERR_NUMERICAL
        return ERR_NONE
    # [P0-3 修复] RecursionError 专项分类
    # RecursionError 是代码逻辑 bug（无限递归），不是数值稳定性问题，
    # BDF/Radau/QSSA 重试策略完全无效，应跳过重试直接返回。
    if re.search(r"RecursionError|Maximum recursion depth", stdout_stderr, re.IGNORECASE):
        return ERR_RECURSION
    # [Sandbox Fix] Permission denied 专项分类（Windows 安全策略拦截）
    if re.search(r"PermissionError|Permission denied|Access is denied", stdout_stderr, re.IGNORECASE):
        return ERR_PERMISSION
    return ERR_RUNTIME


# -----------------------------------------------------------------------------
# [Sandbox Fix] SHA256 校验与 artifact manifest 构建
# -----------------------------------------------------------------------------
def _compute_sha256(file_path: str) -> str:
    """计算文件 SHA256 哈希，用于产物审计。文件不存在时返回空字符串。"""
    try:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError):
        return ""


def _detect_csv_encoding(csv_path: str) -> str:
    """检测 CSV 文件编码（utf-8-sig / utf-8 / ascii）。"""
    try:
        with open(csv_path, "rb") as f:
            raw = f.read(4)
        if raw[:3] == b"\xef\xbb\xbf":
            return "utf-8-sig"
        return "utf-8"
    except (OSError, IOError):
        return "utf-8"


def _build_artifact_manifest(
    work_dir: str,
    csv_path: str,
    stdout: str,
    stderr: str,
) -> dict:
    """构建结构化 artifact manifest，含 SHA256 与 CSV 元数据。

    Args:
        work_dir: 沙箱工作目录（持久化或 TEMP）。
        csv_path: 仿真 CSV 路径（可能不存在）。
        stdout: 子进程 stdout。
        stderr: 子进程 stderr。

    Returns:
        artifact_manifest dict，始终含 csv_path / sha256 / encoding / columns /
        row_count / time_unit / log_path 字段（不存在时为空值）。
    """
    manifest: dict = {
        "csv_path": "",
        "report_path": "",  # sandbox 不产出 report.md，由 report_renderer 节点产出
        "log_path": "",
        "sha256": {},
        "encoding": "utf-8",
        "columns": [],
        "row_count": 0,
        "time_unit": "min",
    }

    # run.log 路径（stdout + stderr 持久化）
    log_path = os.path.join(work_dir, "run.log")
    manifest["log_path"] = log_path

    # CSV 产物. Metadata comes from the shared CSV boundary so encoding and
    # corruption decisions stay identical across execution and evaluation.
    if csv_path and os.path.exists(csv_path):
        manifest["csv_path"] = csv_path
        manifest["sha256"]["simulation.csv"] = _compute_sha256(csv_path)
        try:
            from app.csv_io import read_csv_robust

            csv_result = read_csv_robust(csv_path)
            manifest["encoding"] = csv_result.encoding
            manifest["columns"] = ["time", *csv_result.columns]
            manifest["row_count"] = csv_result.row_count
            if csv_result.error:
                manifest["error"] = csv_result.error
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            logger.warning("artifact manifest CSV 解析失败: %s", exc)

    # PNG 产物
    png_path = os.path.join(work_dir, "simulation.png")
    if os.path.exists(png_path):
        manifest["sha256"]["simulation.png"] = _compute_sha256(png_path)

    # run.py 产物哈希
    run_py_path = os.path.join(work_dir, "run.py")
    if os.path.exists(run_py_path):
        manifest["sha256"]["run.py"] = _compute_sha256(run_py_path)

    return manifest


def _prepare_work_dir(case_id: str | None, artifacts_dir: str | None) -> tuple[str, bool]:
    """准备沙箱工作目录，返回 (work_dir, is_persistent)。

    优先策略：使用 artifacts_dir 持久化路径（非 TEMP），确保产物可审计。
    降级策略：artifacts_dir 不可用时回退到 TEMP，并记录 warning。

    Args:
        case_id: 用例标识（用于子目录命名）。
        artifacts_dir: 持久化 artifacts 根目录。

    Returns:
        (work_dir 绝对路径, is_persistent: True 表示持久化目录不清理)
    """
    if artifacts_dir:
        # 持久化路径：artifacts_dir / {case_id or 'default'} / sandbox_run
        sub_dir = case_id if case_id else "default"
        work_dir = os.path.join(artifacts_dir, sub_dir, "sandbox_run")
        try:
            os.makedirs(work_dir, exist_ok=True)
            # 验证可写性
            test_file = os.path.join(work_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            return work_dir, True
        except (OSError, PermissionError) as exc:
            logger.warning(
                "[Sandbox Fix] 持久化工作目录创建失败 (%s): %s，降级到 TEMP",
                work_dir, exc,
            )

    # 降级：TEMP 目录
    temp_dir = tempfile.mkdtemp(prefix="bio_dynamics_v2_")
    logger.warning(
        "[Sandbox Fix] 使用 TEMP 降级目录: %s (case_id=%s)", temp_dir, case_id
    )
    return temp_dir, False


def _persist_run_log(work_dir: str, stdout: str, stderr: str) -> None:
    """将子进程 stdout/stderr 持久化到 work_dir/run.log，供审计复核。"""
    try:
        log_path = os.path.join(work_dir, "run.log")
        with open(log_path, "w", encoding="utf-8") as f:
            if stdout:
                f.write("=== STDOUT ===\n")
                f.write(stdout)
                if not stdout.endswith("\n"):
                    f.write("\n")
            if stderr:
                f.write("=== STDERR ===\n")
                f.write(stderr)
                if not stderr.endswith("\n"):
                    f.write("\n")
    except (OSError, PermissionError) as exc:
        logger.warning("[Sandbox Fix] run.log 写入失败: %s", exc)


def execute_simulation_code_v2(
    code: str,
    timeout: int | None = None,
    allow_stochastic: bool = False,
    case_id: str | None = None,
    artifacts_dir: str | None = None,
) -> dict:
    """v2 沙箱执行入口：AST 预检 → 安全扫描 → 确定性检查 → 执行 → 错误分类 → CSV/PNG 收集。

    与 v1 的区别：
    1. 前置 ast.parse() 拦截语法错误，避免进入 subprocess 浪费 60s
    2. 错误分类写入 error_class 字段
    3. 始终返回 simulation_csv_path（若代码生成 CSV）
    4. 安全拦截写入 error_class=security_error
    5. 深度审核报告 §3.1：确定性求解检查（默认禁随机噪声）+ 审计日志 + metrics 埋点

    [Sandbox Fix] 新增持久化工作目录与结构化错误返回：
    6. case_id + artifacts_dir 参数化产物路径（非 TEMP），重启后仍可审计
    7. 返回 execution_status / stdout / stderr / artifact_manifest 结构化字段
    8. SHA256 校验记录到 artifact_manifest.sha256
    9. 降级 fallback：持久化路径不可用时回退 TEMP + warning
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
            "execution_status": "failed",
            "stdout_stderr": ast_msg,
            "stdout": "",
            "stderr": ast_msg,
            "image_base64": "",
            "error_class": ERR_SYNTAX,
            "simulation_csv_path": "",
            "artifact_manifest": {},
        }

    # 2. 静态安全扫描
    is_safe, security_msg = _check_code_security(code)
    if not is_safe:
        duration = time.perf_counter() - exec_start
        _write_audit_log(code, "error", ERR_SECURITY, duration, security_msg)
        metrics.record_sandbox_execution(False, ERR_SECURITY, duration)
        return {
            "status": "error",
            "execution_status": "failed",
            "stdout_stderr": security_msg,
            "stdout": "",
            "stderr": security_msg,
            "image_base64": "",
            "error_class": ERR_SECURITY,
            "simulation_csv_path": "",
            "artifact_manifest": {},
        }

    # 3. 确定性求解检查（深度审核报告 §3.1）
    det_ok, det_msg = _check_determinism(code, allow_stochastic=allow_stochastic)
    if not det_ok:
        duration = time.perf_counter() - exec_start
        _write_audit_log(code, "error", ERR_SECURITY, duration, det_msg)
        metrics.record_sandbox_execution(False, ERR_SECURITY, duration)
        return {
            "status": "error",
            "execution_status": "failed",
            "stdout_stderr": det_msg,
            "stdout": "",
            "stderr": det_msg,
            "image_base64": "",
            "error_class": ERR_SECURITY,
            "simulation_csv_path": "",
            "artifact_manifest": {},
        }

    # 4. 执行 —— [Sandbox Fix] 优先使用持久化工作目录
    work_dir, is_persistent = _prepare_work_dir(case_id, artifacts_dir)
    try:
        cleaned = code.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
        cleaned = re.sub(r"\bplt\.show\(\s*\)", "", cleaned)

        # [Sandbox N1] 注入 import guard bootstrap 到 run.py 头部
        # 在用户代码执行前安装 import 拦截器，强制白名单导入
        bootstrap_code = _build_import_guard_bootstrap(work_dir)

        run_path = os.path.join(work_dir, "run.py")
        with open(run_path, "w", encoding="utf-8") as f:
            if bootstrap_code:
                f.write(bootstrap_code)
            f.write(cleaned)

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        # Keep subprocess streams and regular text files UTF-8 on Windows.
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # [BM2-BM8 修复 / Mode B] v4_ode_system 可能 import app.solvers.dde_solver 等
        # 模块，sandbox 临时目录不包含 app 包。将 backend 根目录加入 PYTHONPATH，
        # 使 sandbox 内可 import app.* 模块（仅 v4 ODE 模板需要，v3 模板不受影响）。
        # [Sandbox N1] PYTHONPATH 仍保留 backend_root（dde_solver 机械可导入），
        # 但 import guard 会严格拦截除 app.solvers.dde_solver 外的所有 app.* 子模块，
        # 防止子进程读取 app.config (API key) / app.sandbox / app.main 等敏感模块。
        backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = backend_root + os.pathsep + env["PYTHONPATH"]
        else:
            env["PYTHONPATH"] = backend_root

        # [Sandbox N1] 网络隔离（best-effort）：阻止 HTTP 代理出站
        # socket 模块被 import guard 拦截，网络访问天然被阻断；
        # NO_PROXY / no_proxy=* 进一步阻止 requests / urllib 通过代理出站
        env["NO_PROXY"] = "*"
        env["no_proxy"] = "*"
        # 显式清空代理环境变量，防止子进程通过环境变量配置的代理出站
        for _proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
            env.pop(_proxy_var, None)

        # [Sandbox N1] 资源限制：Unix 用 preexec_fn 设置 RLIMIT_*，Windows 降级
        preexec_fn = _build_preexec_fn()
        if preexec_fn is None and sys.platform != "win32":
            logger.warning(
                "[Sandbox N1] 资源限制未启用（platform=%s），子进程无内存/CPU/文件大小限制",
                sys.platform,
            )
        elif sys.platform == "win32":
            logger.info(
                "[Sandbox N1] Windows 平台不支持 preexec_fn 资源限制，降级运行"
                "（依赖 import guard + 超时 kill 保证安全）"
            )

        was_timeout = False
        stdout_text = ""
        stderr_text = ""
        returncode = -1
        try:
            # [Sandbox N1] 使用 Popen + 升级 kill 替代 subprocess.run
            # 超时后 SIGTERM → SIGKILL（Unix）/ taskkill /F /T /PID（Windows）递归 kill 子进程树
            returncode, stdout_text, stderr_text, was_timeout = _run_subprocess_with_escalation(
                cmd=[sys.executable, "run.py"],
                cwd=work_dir,
                env=env,
                timeout=timeout,
                preexec_fn=preexec_fn,
            )
        except PermissionError as exc:
            # [Sandbox Fix] Windows 安全策略拒绝执行
            duration = time.perf_counter() - exec_start
            err_msg = f"沙箱执行权限拒绝 (PermissionError): {exc}"
            _persist_run_log(work_dir, "", err_msg)
            _write_audit_log(code, "error", ERR_PERMISSION, duration, err_msg)
            metrics.record_sandbox_execution(False, ERR_PERMISSION, duration)
            return {
                "status": "error",
                "execution_status": "failed",
                "stdout_stderr": err_msg,
                "stdout": "",
                "stderr": err_msg,
                "image_base64": "",
                "error_class": ERR_PERMISSION,
                "simulation_csv_path": "",
                "artifact_manifest": _build_artifact_manifest(work_dir, "", "", err_msg),
            }
        except Exception as exc:
            duration = time.perf_counter() - exec_start
            err_msg = f"沙箱执行异常: {exc}"
            _persist_run_log(work_dir, stdout_text, stderr_text + "\n" + err_msg)
            _write_audit_log(code, "error", ERR_RUNTIME, duration, err_msg)
            metrics.record_sandbox_execution(False, ERR_RUNTIME, duration)
            return {
                "status": "error",
                "execution_status": "failed",
                "stdout_stderr": err_msg,
                "stdout": stdout_text,
                "stderr": stderr_text + "\n" + err_msg,
                "image_base64": "",
                "error_class": ERR_RUNTIME,
                "simulation_csv_path": "",
                "artifact_manifest": _build_artifact_manifest(work_dir, "", stdout_text, stderr_text),
            }

        # [Sandbox N1] 超时分类（Popen 升级 kill 路径）
        if was_timeout:
            duration = time.perf_counter() - exec_start
            timeout_msg = f"代码执行超过 {timeout} 秒，已被终止。"
            _persist_run_log(work_dir, stdout_text, stderr_text + "\n" + timeout_msg)
            _write_audit_log(code, "error", ERR_TIMEOUT, duration, timeout_msg)
            metrics.record_sandbox_execution(False, ERR_TIMEOUT, duration)
            return {
                "status": "error",
                "execution_status": "timeout",
                "stdout_stderr": timeout_msg,
                "stdout": stdout_text,
                "stderr": stderr_text + "\n" + timeout_msg,
                "image_base64": "",
                "error_class": ERR_TIMEOUT,
                "simulation_csv_path": "",
                "artifact_manifest": _build_artifact_manifest(work_dir, "", stdout_text, stderr_text),
            }

        # [Sandbox N1] import guard 拦截分类
        # 用户代码尝试 import 非白名单模块（os / sys / socket / app.config ...）
        # 时，import guard 抛 _ImportGuardError，stderr 含 [ImportBlocked] 签名
        if _detect_import_blocked(stderr_text):
            duration = time.perf_counter() - exec_start
            blocked_msg = (
                "安全拦截：import guard 阻止非白名单模块导入。\n"
                f"stderr 摘要: {stderr_text[-500:] if stderr_text else '(empty)'}"
            )
            _persist_run_log(work_dir, stdout_text, stderr_text)
            _write_audit_log(code, "error", ERR_IMPORT_BLOCKED, duration, blocked_msg)
            metrics.record_sandbox_execution(False, ERR_IMPORT_BLOCKED, duration)
            return {
                "status": "error",
                "execution_status": "failed",
                "stdout_stderr": blocked_msg,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "image_base64": "",
                "error_class": ERR_IMPORT_BLOCKED,
                "simulation_csv_path": "",
                "artifact_manifest": _build_artifact_manifest(work_dir, "", stdout_text, stderr_text),
            }

        stdout_stderr = stdout_text
        if stderr_text:
            if stdout_stderr:
                stdout_stderr += "\n"
            stdout_stderr += stderr_text

        # 持久化 run.log（stdout + stderr）
        _persist_run_log(work_dir, stdout_text, stderr_text)

        # 5. 收集产物
        image_base64 = ""
        image_path = os.path.join(work_dir, "simulation.png")
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        csv_path = os.path.join(work_dir, "simulation.csv")
        simulation_csv_path = csv_path if os.path.exists(csv_path) else ""

        # [Sandbox Fix] 持久化目录无需复制；TEMP 降级目录需复制到持久化审计目录
        if simulation_csv_path and not is_persistent:
            # 降级场景：尝试复制到 SANDBOX_AUDIT_LOG_DIR 下的 csv 子目录
            try:
                audit_csv_dir = os.path.join(settings.SANDBOX_AUDIT_LOG_DIR, "csv_fallback")
                os.makedirs(audit_csv_dir, exist_ok=True)
                fallback_name = f"simulation_{case_id or 'default'}.csv"
                fallback_csv = os.path.join(audit_csv_dir, fallback_name)
                shutil.copy2(simulation_csv_path, fallback_csv)
                simulation_csv_path = fallback_csv
                logger.warning(
                    "[Sandbox Fix] TEMP 降级场景，CSV 已复制到审计目录: %s",
                    fallback_csv,
                )
            except (OSError, PermissionError) as exc:
                logger.warning(
                    "[Sandbox Fix] CSV 复制到审计目录失败: %s", exc
                )

        # 6. 错误分类
        error_class = _classify_error(
            returncode=returncode,
            stdout_stderr=stdout_stderr,
            was_timeout=was_timeout,
        )
        status = (
            "success"
            if error_class == ERR_NONE and (image_base64 or simulation_csv_path)
            else "error"
        )

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

        # 8. 构建 artifact manifest（含 SHA256）
        artifact_manifest = _build_artifact_manifest(
            work_dir, simulation_csv_path, stdout_text, stderr_text
        )

        # 9. 审计日志 + metrics 埋点（深度审核报告 §3.1 + §3.3）
        duration = time.perf_counter() - exec_start
        _write_audit_log(
            code, status, error_class, duration, stdout_stderr,
            metadata={
                "returncode": returncode,
                "csv_path": simulation_csv_path,
                "case_id": case_id,
                "work_dir": work_dir,
                "is_persistent": is_persistent,
            },
        )
        metrics.record_sandbox_execution(status == "success", error_class, duration)

        # [Sandbox Fix] 结构化 execution_status（与 status 对齐，但独立字段供 orchestrator 判断）
        execution_status = "success" if status == "success" else "failed"

        return {
            "status": status,
            "execution_status": execution_status,
            "stdout_stderr": stdout_stderr,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "image_base64": image_base64,
            "error_class": error_class,
            "simulation_csv_path": simulation_csv_path,
            "returncode": returncode,
            "duration_seconds": round(duration, 4),
            "artifact_manifest": artifact_manifest,
            "work_dir": work_dir,
            "is_persistent": is_persistent,
        }
    finally:
        # [Sandbox Fix] 持久化目录不清理（保留产物供审计）；仅清理 TEMP 降级目录
        if not is_persistent:
            shutil.rmtree(work_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# [v5 Recovery Sprint 3 / RC6] 数值稳定性重试包装
# 旧实现：LSODA 崩溃后直接返回错误（62.5% 崩溃率），numerical_stability_retry.py
# 模块存在但从未被调用。刚性系统（p53/NF-κB 快慢时间尺度）无 BDF/Radau 回退。
# 修复：包装 execute_simulation_code_v2，失败时按阶梯策略（收紧步长→BDF→Radau→QSSA）
# 修改代码并重试，最多 4 次。
# -----------------------------------------------------------------------------
def execute_with_stability_retry(
    code: str,
    timeout: int | None = None,
    allow_stochastic: bool = False,
    max_retries: int = 4,
    case_id: str | None = None,
    artifacts_dir: str | None = None,
) -> dict:
    """[v5 RC6] 带 BDF/Radau 数值稳定性重试的沙箱执行。

    当 LSODA 求解器崩溃（runtime_error / numerical_error）时，自动按阶梯策略
    修改求解器参数并重试：收紧 max_step → BDF → Radau → QSSA 降阶。

    Args:
        code: ODE 仿真代码
        timeout: 沙箱超时（秒）
        allow_stochastic: 是否允许随机噪声
        max_retries: 最大重试次数（默认 4，与 NumericalStabilityRetry.STRATEGIES 一致）
        case_id: [Sandbox Fix] 用例标识，用于参数化产物持久化路径
        artifacts_dir: [Sandbox Fix] 持久化 artifacts 根目录（非 TEMP）

    Returns:
        首次成功的执行结果，或最后一次失败的执行结果（含 stability_retry_exhausted=True）
    """
    from app.solvers.numerical_stability_retry import NumericalStabilityRetry

    retryer = NumericalStabilityRetry()
    current_code = code
    attempt = 1
    last_result: dict = {}

    while True:
        result = execute_simulation_code_v2(
            current_code,
            timeout=timeout,
            allow_stochastic=allow_stochastic,
            case_id=case_id,
            artifacts_dir=artifacts_dir,
        )
        if result["status"] == "success":
            if attempt > 1:
                result["stability_retry_applied"] = True
                result["stability_retry_attempts"] = attempt - 1
                logger.info(
                    "[v5 RC6] 数值稳定性重试成功（attempt=%d）", attempt - 1
                )
            return result

        # 仅对 runtime_error / numerical_error 重试（不重试 syntax/security/timeout/recursion）
        if result.get("error_class") not in (ERR_RUNTIME, ERR_NUMERICAL):
            return result

        # [P0-3 修复] RecursionError 是代码逻辑 bug，BDF/Radau/QSSA 策略无效，直接返回
        if result.get("error_class") == ERR_RECURSION:
            logger.warning(
                "[P0-3] RecursionError 检测到，跳过数值稳定性重试（逻辑 bug 不可恢复）"
            )
            return result

        if attempt > max_retries:
            result["stability_retry_exhausted"] = True
            logger.warning(
                "[v5 RC6] 数值稳定性重试已用尽所有策略（%d 次），返回最后失败结果",
                max_retries,
            )
            return result

        retry_result = retryer.retry(
            current_code,
            result.get("error_class", "unknown"),
            attempt,
        )
        if retry_result["exhausted"]:
            result["stability_retry_exhausted"] = True
            return result

        current_code = retry_result["modified_code"]
        logger.info(
            "[v5 RC6] 应用策略 %s（attempt=%d），重试执行",
            retry_result["strategy"],
            attempt,
        )
        attempt = retry_result["next_attempt"]
        last_result = result


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

    # 1. 使用统一 CSV 边界，确保 UTF-8-SIG/GB18030 与 evaluator 行为一致。
    from app.csv_io import read_csv_robust

    csv_result = read_csv_robust(csv_path)
    if csv_result.empty:
        logger.warning(
            "post_simulation_validation 读取 CSV 失败: %s (%s)",
            csv_path,
            csv_result.error,
        )
        result["passed"] = False
        result["error"] = csv_result.error or "empty_csv"
        return result

    # 3. 筛选 CSV 中实际存在的物种列
    # TD-003 接线：species_names 为 None 时自动从 CSV 列头检测（排除时间列）
    if species_names is None:
        available_species = list(csv_result.species)
        logger.debug(
            "post_simulation_validation 自动检测物种列: %s", available_species
        )
    else:
        available_species = [s for s in species_names if s in csv_result.species]
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
    for row_index, time_value in enumerate(csv_result.times):
        for species in available_species:
            trajectory = csv_result.species[species]
            if row_index >= len(trajectory):
                continue
            value = trajectory[row_index]
            if math.isnan(value):
                result["nan_detected"] = True
                continue
            if math.isinf(value):
                result["inf_detected"] = True
                continue

            species_data[species].append((time_value, value))
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
