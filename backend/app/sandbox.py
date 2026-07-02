# BioDynamics Agent - 代码执行沙箱
# 使用 subprocess 在隔离临时目录中运行 LLM 生成的 Python 仿真代码，
# 捕获控制台输出并将生成的图片转为 Base64。
# 新增：静态安全扫描（拦截危险模块导入）与生物学常识检查（负浓度 / NaN / Inf）。

import base64
import os
import re
import shutil
import subprocess
import tempfile


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
                timeout=30,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "stdout_stderr": "代码执行超过 30 秒，已被终止。",
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

        return {
            "status": status,
            "stdout_stderr": stdout_stderr,
            "image_base64": image_base64,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
