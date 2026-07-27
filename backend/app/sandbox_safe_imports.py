# BioDynamics Agent - Sandbox Import Guard
#
# 在 sandbox 子进程启动时注入 run.py 头部，通过覆盖 builtins.__import__
# 实现严格白名单导入拦截：
#   - SAFE_MODULES         : 允许的第三方/标准库根模块（numpy / scipy / ...）
#   - SAFE_APP_SUBMODULES  : 允许的 app.* 子模块（v4 ODE 模板需要 dde_solver）
#   - BLOCKED_MODULES      : 高危模块显式黑名单（仅用于错误消息清晰度）
#
# 拦截策略：
#   1. 用户代码（caller=__main__）只能 import 白名单内的模块
#   2. 白名单模块自身的 transitive imports 自动放行（通过 caller frame 检测）
#      例如 numpy 内部 import os 不会被拦截，但用户代码 import os 会被拦截
#   3. app.* 只允许 SAFE_APP_SUBMODULES 中显式列出的子模块
#   4. 拦截时抛 ImportError，消息含 [ImportBlocked] 签名，sandbox.py 据此分类
#
# 设计权衡：
#   - meta_path finder 只在模块不在 sys.modules 时被调用，无法拦截 os/sys 等
#     启动时已加载的模块；因此选择覆盖 builtins.__import__（每次 import 都调用）
#   - 通过 caller frame 判断"是否白名单模块内部的 import"，避免阻断 numpy/scipy
#     的 transitive 依赖加载
#   - 不替代 sandbox.py 的 AST 黑名单（_BLOCKED_MODULES），而是 defense-in-depth：
#     AST 拦直接 import 语句；本 guard 拦所有 import 调用（含动态 import）

from __future__ import annotations

import builtins
import sys
from typing import Any

# -----------------------------------------------------------------------------
# 白名单 / 黑名单定义
# -----------------------------------------------------------------------------
# 用户代码（caller 非白名单）允许 import 的根模块。
# 子模块自动允许（如 numpy.linalg / scipy.integrate / matplotlib.pyplot）。
SAFE_MODULES = frozenset({
    # 数值/科学计算（第三方）
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    # 标准库 - 纯计算/数据结构（无 IO/网络/进程）
    "math",
    "random",
    "collections",
    "itertools",
    "functools",
    "typing",
    "dataclasses",
    "enum",
    "abc",
    "copy",
    "datetime",
    "json",       # 只读解析（用户无法 open 文件，AST 拦截 open）
    "re",
    "numbers",
    "decimal",
    "fractions",
    "statistics",
    "string",
    "textwrap",
    "unicodedata",
    "pprint",
    "reprlib",
    "time",       # 纯时间函数（time.time/sleep/perf_counter），无文件/网络/进程访问
    # 日志/警告：用户 ODE 代码偶尔需要，且无文件/网络副作用（AST 已拦 open）
    "logging",
    "warnings",
})

# 高危模块显式黑名单。仅用于生成更清晰的错误消息；
# 实际拦截靠白名单（任何不在 SAFE_MODULES / SAFE_APP_SUBMODULES 的模块都被拦）。
BLOCKED_MODULES = frozenset({
    "os",
    "sys",
    "subprocess",
    "socket",
    "http",
    "urllib",
    "threading",
    "multiprocessing",
    "ctypes",
    "importlib",
    "asyncio",
    "selectors",
    "signal",
    "pickle",
    "marshal",
    "shelve",
    "pathlib",
    "shutil",
    "tempfile",
    "glob",
    "fnmatch",
    "ftplib",
    "smtplib",
    "email",
    "poplib",
    "imaplib",
    "telnetlib",
    "xmlrpc",
    "builtins",
    "requests",
    "urllib3",
    "httpx",
    "aiohttp",
})

# 允许的 app.* 子模块（v4 ODE 模板需要 DDE 求解器）。
# 任何其他 app.* 子模块（app.config / app.sandbox / app.main ...）都被拦截，
# 防止 sandbox 子进程读取 API key 或调用内部服务。
SAFE_APP_SUBMODULES = frozenset({
    "app.solvers.dde_solver",
})

# 危险的内置 C 模块子集（sys.builtin_module_names 中需显式拦截的模块）。
# 这些模块提供直接的文件/进程/网络/注册表/运行时内部访问，即使用户代码
# 直接 import 也应拦截。未列出的内置 C 模块（_io/_thread/_collections 等）
# 在已加载状态下允许 re-import（仅 sys.modules 缓存查找，无新代码执行）。
#
# 背景：numpy/scipy 的 C 扩展在初始化时通过 PyImport_Import 调用
# __import__("_io") 等，此时 C 代码无 Python 帧，栈遍历无法识别为
# 白名单 transitive import。这些内置 C 模块在 Python 启动时已加载，
# re-import 是无副作用的缓存查找，安全放行。
_DANGEROUS_BUILTIN_C_MODULES = frozenset({
    # 系统/运行时内部（可绕过 guard 或干扰清理）
    "sys",
    "builtins",
    "_imp",                # import 机器内部（可动态加载任意模块）
    "_xxsubinterpreters",  # 子解释器管理（可逃逸沙箱）
    "gc",                  # 垃圾回收器控制（可导致崩溃）
    "faulthandler",        # 故障处理器（可泄露内存状态）
    "_tracemalloc",        # 内存分配追踪（信息泄露）
    "atexit",              # 退出回调（可在退出时执行任意代码）
    # 文件/进程/注册表访问
    "nt",                  # Windows NT API（低级文件/进程，os 模块底层）
    "_winapi",             # Windows API（文件/进程/窗口操作）
    "msvcrt",              # MS C 运行时（控制台 I/O、进程控制）
    "mmap",                # 内存映射文件（文件访问）
    "winreg",              # Windows 注册表（配置篡改）
    # 序列化（代码执行风险）
    "marshal",             # marshal 反序列化可执行任意代码
})

# 拦截签名：sandbox.py 在 stderr 中搜索此字符串以分类为 ERR_IMPORT_BLOCKED
_BLOCKED_SIGNAL = "[ImportBlocked]"


class _ImportGuardError(ImportError):
    """导入被拦截时抛出的异常。

    继承 ImportError，因此用户代码中的 try/except ImportError 仍可捕获，
    不会导致意外的进程崩溃（例如 dde_solver.py 的 jitcdde try-import）。
    """


# -----------------------------------------------------------------------------
# Caller frame 白名单检测
# -----------------------------------------------------------------------------
def _is_caller_whitelisted(caller_module: str) -> bool:
    """判断调用 import 的模块自身是否在白名单内。

    白名单模块（numpy / scipy / app.solvers.dde_solver ...）内部的 import
    自动放行，避免阻断其 transitive 依赖加载。

    Args:
        caller_module: 调用 __import__ 的帧的 __name__（如 "numpy" / "__main__"）

    Returns:
        True 如果 caller 自身在白名单内（其 import 自动放行）
    """
    if not caller_module:
        return False
    root = caller_module.split(".", 1)[0]
    if root in SAFE_MODULES:
        return True
    # app.* 白名单子模块
    for safe in SAFE_APP_SUBMODULES:
        if caller_module == safe or caller_module.startswith(safe + "."):
            return True
    return False


def _is_import_triggered_by_whitelist() -> bool:
    """遍历整个调用栈，检查是否有任意一帧属于白名单模块。

    解决 C 扩展 transitive import 问题：
        当 numpy 的 C 扩展（multiarray.pyd）调用 PyImport_Import("_io") 时，
        sys._getframe(1) 返回的是触发 C 扩展加载的 Python 帧（通常是 __main__），
        而非 C 扩展本身（C 代码无 Python 帧）。
        单帧检测会误判为 "用户代码 import _io" 并拦截，导致 numpy 崩溃。

    栈遍历方案：
        用户代码 import numpy → numpy.__init__ 执行 → C 扩展加载 → import _io
        此时调用栈为: __main__ → numpy.__init__ → (C, 无帧) → __import__ → guard
        遍历栈可发现 "numpy" 帧（白名单），正确判定为 transitive import。

    而用户直接 import _io 的栈为: __main__ → __import__ → guard
        遍历栈只发现 "__main__"（非白名单），正确拦截。

    Returns:
        True 如果调用栈中任意一帧属于白名单模块
    """
    try:
        frame = sys._getframe(1)  # 从 __import__ 的直接调用者开始
        while frame is not None:
            module_name = frame.f_globals.get("__name__", "") if frame.f_globals else ""
            if module_name and _is_caller_whitelisted(module_name):
                return True
            frame = frame.f_back
    except (ValueError, AttributeError):
        pass
    return False


def _check_import_allowed(name: str, caller_module: str) -> None:
    """检查 import 是否允许，不允许则抛 _ImportGuardError。

    Args:
        name: 待 import 的模块全名（如 "numpy" / "os" / "app.solvers.dde_solver"）
        caller_module: 调用 __import__ 的帧的 __name__

    Raises:
        _ImportGuardError: 如果 import 不在白名单内
    """
    if not name:
        return
    root = name.split(".", 1)[0]

    # 1. 白名单根模块（numpy / scipy / math ...）及其子模块一律允许
    if root in SAFE_MODULES:
        return

    # 2. app.* 只允许显式列出的白名单子模块
    if root == "app":
        for safe in SAFE_APP_SUBMODULES:
            if name == safe or name.startswith(safe + "."):
                return
        raise _ImportGuardError(
            f"{_BLOCKED_SIGNAL} module='{name}' "
            f"(app.* not in whitelist; allowed: {sorted(SAFE_APP_SUBMODULES)})"
        )

    # 3. 显式黑名单（错误消息更清晰）
    if root in BLOCKED_MODULES:
        raise _ImportGuardError(
            f"{_BLOCKED_SIGNAL} module='{name}' "
            f"(root '{root}' is blocked)"
        )

    # 4. 未知模块（既不在白名单也不在黑名单）→ 默认拒绝（allowlist 模型）
    raise _ImportGuardError(
        f"{_BLOCKED_SIGNAL} module='{name}' "
        f"(root '{root}' not in whitelist)"
    )


# -----------------------------------------------------------------------------
# __import__ 覆盖安装
# -----------------------------------------------------------------------------
_INSTALLED = False
_ORIGINAL_IMPORT: Any = None


def install_import_guard() -> None:
    """安装 import 拦截器（覆盖 builtins.__import__）。

    幂等：重复调用为 no-op。

    拦截逻辑：
        1. 解析相对 import（level > 0）到绝对名
        2. 通过 sys._getframe(1) 获取 caller 模块名
        3. 如果 caller 自身在白名单内（如 numpy 内部 import）→ 放行
        4. 否则按白名单检查 import 目标 → 拒绝则抛 _ImportGuardError
        5. 允许则委托给原始 __import__
    """
    global _INSTALLED, _ORIGINAL_IMPORT
    if _INSTALLED:
        return
    _ORIGINAL_IMPORT = builtins.__import__

    def _guarded_import(name: str, globals: Any = None, locals: Any = None,
                        fromlist: tuple = (), level: int = 0) -> Any:
        # 相对 import 解析（sandbox 用户代码通常为 __main__，无相对 import，
        # 但 dde_solver.py 等模块内部可能使用）
        resolved_name = name
        if level > 0:
            caller_globals = globals or {}
            caller_package = caller_globals.get("__package__", "") or ""
            if caller_package:
                parts = caller_package.split(".")
                # level=1 → 当前 package；level=2 → 父 package ...
                if level - 1 >= len(parts):
                    base = parts[0] if parts else ""
                else:
                    base = ".".join(parts[: -(level - 1)] if level > 1 else parts)
                resolved_name = (base + "." + name) if name else base
            else:
                resolved_name = name

        # [Sandbox N1 Fix] 内置 C 模块已加载放行：
        # numpy/scipy 的 C 扩展在初始化时通过 PyImport_Import 调用
        # __import__("_io") 等，C 代码无 Python 帧，栈遍历可能失败。
        # 但这些内置 C 模块在 Python 启动时已加载（sys.modules 缓存查找，
        # 无新代码执行），且不属于危险子集，安全放行。
        root = resolved_name.split(".", 1)[0]
        if (root in sys.modules
                and root in sys.builtin_module_names
                and root not in _DANGEROUS_BUILTIN_C_MODULES):
            return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

        # [Sandbox N1 Fix] 栈遍历检测：白名单模块的 transitive import 自动放行
        # 遍历整个调用栈而非单帧，解决 C 扩展（numpy multiarray 等）调用
        # PyImport_Import 时无 Python 帧导致单帧检测失败的问题。
        # 例：用户 import numpy → numpy.__init__ → C ext → import _io
        #     栈中存在 "numpy" 帧 → 判定为 transitive → 放行 _io
        if _is_import_triggered_by_whitelist():
            return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

        # 用户代码 import → 严格白名单检查
        # 获取 caller 模块名仅用于错误消息，不影响放行决策
        try:
            frame = sys._getframe(1)
            caller_module = frame.f_globals.get("__name__", "") if frame else ""
        except (ValueError, AttributeError):
            caller_module = ""

        _check_import_allowed(resolved_name, caller_module)

        # 允许 → 委托原始 __import__
        return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

    builtins.__import__ = _guarded_import
    _INSTALLED = True


def uninstall_import_guard() -> None:
    """恢复原始 __import__（仅供测试使用）。"""
    global _INSTALLED, _ORIGINAL_IMPORT
    if not _INSTALLED:
        return
    if _ORIGINAL_IMPORT is not None:
        builtins.__import__ = _ORIGINAL_IMPORT
    _INSTALLED = False
    _ORIGINAL_IMPORT = None


def is_guard_installed() -> bool:
    """返回 guard 是否已安装（仅供测试/诊断使用）。"""
    return _INSTALLED
