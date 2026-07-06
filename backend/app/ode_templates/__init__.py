"""BioDynamics Agent - ODE 模板加载器（v2 N6）

使用 Jinja2 加载 `app/ode_templates/*.j2`。
StrictUndefined 保证缺失变量立即失败，避免静默错误。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATE_DIR = Path(__file__).parent
_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_template(name: str, vars: dict[str, Any]) -> str:
    """渲染指定模板（无 .j2 后缀）。"""
    return _ENV.get_template(f"{name}.j2").render(**(vars or {}))


def list_templates() -> list[str]:
    """列出所有可用模板名（无后缀）。"""
    return [p.stem for p in sorted(_TEMPLATE_DIR.glob("*.j2"))]
