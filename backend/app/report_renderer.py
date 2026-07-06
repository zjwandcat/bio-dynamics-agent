"""BioDynamics Agent - 报告渲染器（v2 N11）

职责：
- LLM 只输出 4 个字符串字段（mechanism_analysis / simulation_interpretation /
  discussion / limitations），renderer 用 Jinja2 模板把它们与 metrics / evidence /
  experiments / knowledge_graph 拼装成最终 Markdown。
- 强校验：禁止使用 "某疾病" / "T1" / "T2" / "炎症因子" 等模糊量词，
  命中则返回 violations，由 N11 触发 LLM 重试一次。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

# 默认模板目录：与 renderer 同包
_DEFAULT_TEMPLATE_DIR = Path(__file__).parent / "report_templates"


class ReportRenderer:
    """Markdown 报告渲染器。"""

    # 严格禁止的措辞：命中后 N11 触发 LLM 重试
    FORBIDDEN_TERMS: list[str] = [
        "某疾病",       # vague placeholder
        "T1", "T2",     # vague placeholders
        "炎症因子",     # catch-all filler
        "等等", "等",   # catch-all filler
        "TGF-betta",    # known misspelling
    ]

    def __init__(self, template_dir: str | Path | None = None) -> None:
        tpl_dir = Path(template_dir) if template_dir else _DEFAULT_TEMPLATE_DIR
        self.env = Environment(
            loader=FileSystemLoader(str(tpl_dir)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template_dir = tpl_dir

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def check_forbidden_terms(self, llm_filled: dict[str, Any]) -> list[str]:
        """扫描 LLM 输出，若包含禁止词则返回违例列表。"""
        violations: list[str] = []
        if not isinstance(llm_filled, dict):
            return ["llm_filled is not a dict"]
        for key, value in llm_filled.items():
            text = str(value or "")
            for term in self.FORBIDDEN_TERMS:
                if term in text:
                    violations.append(f"{key} 包含禁止词：{term!r}")
        return violations

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(
        self,
        llm_filled: dict[str, Any],
        metrics: dict[str, Any],
        evidence: list[dict],
        experiments: list[dict],
        knowledge_graph: dict,
        confidence: float | None,
        sandbox_failure_reason: str = "",
        template_name: str = "standard.md.j2",
        time_unit: str = "min",
    ) -> str:
        """用 Jinja2 模板拼装最终 Markdown。

        Args:
            llm_filled: 4 个 LLM 字段（mechanism_analysis 等）。
            metrics: 来自 N8 的指标字典。
            evidence: N10 的文献证据列表。
            experiments: N9 的实验方案列表。
            knowledge_graph: N4 的 KG。
            confidence: 整体置信度（None 表示仿真失败）。
            sandbox_failure_reason: 仿真失败原因描述。
            template_name: 模板文件名（默认 standard.md.j2）。
            time_unit: 时间单位（"min" / "s" / "h"），透传到模板表头。

        Returns:
            渲染好的 Markdown 字符串。
        """
        template = self.env.get_template(template_name)
        return template.render(
            llm=llm_filled or {},
            metrics=metrics or {"species": {}, "overall": {}, "combo": {}},
            evidence=list(evidence or []),
            experiments=list(experiments or []),
            knowledge_graph=knowledge_graph or {},
            confidence=confidence,
            sandbox_failure_reason=sandbox_failure_reason or "",
            time_unit=time_unit,
        )
