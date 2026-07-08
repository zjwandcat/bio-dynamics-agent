"""验证报告生成器 — Verification Report Generator

从 pytest 输出（_collected.jsonl + summary.json）聚合生成三种格式报告：
  - JSON   : machine-readable，含每个用例的指标与状态
  - Markdown: human-readable，含分类统计表与失败用例列表
  - HTML   : 带 CSS 样式的可视化报告，含图表占位

用法：
    python generate_report.py [--results-dir reports/] [--output-dir reports/]

输入：
    reports/_collected.jsonl   每行一个用例结果（pytest_runtest_makereport 钩子写入）
    reports/summary.json       会话级聚合统计（pytest_sessionfinish 钩子写入）

输出：
    reports/report.json
    reports/report.md
    reports/report.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def load_collected(results_dir: Path) -> list[dict[str, Any]]:
    """加载 _collected.jsonl 每行一个用例结果。"""
    collected_path = results_dir / "_collected.jsonl"
    if not collected_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with collected_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_summary(results_dir: Path) -> dict[str, Any]:
    """加载 summary.json 会话级聚合统计。"""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# --------------------------------------------------------------------------- #
# 聚合统计
# --------------------------------------------------------------------------- #
def aggregate_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按 outcome / marker / 文件聚合统计。"""
    by_outcome: Counter[str] = Counter()
    by_file: dict[str, Counter[str]] = defaultdict(Counter)
    by_marker: Counter[str] = Counter()
    by_pathway: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []

    for r in records:
        outcome = r.get("outcome", "unknown")
        by_outcome[outcome] += 1

        nodeid = r.get("nodeid", "")
        file_path = nodeid.split("::")[0] if "::" in nodeid else nodeid
        by_file[file_path][outcome] += 1

        markers = r.get("markers", [])
        for m in markers:
            by_marker[m] += 1

        # 通路分类（从 nodeid 中提取 TestXxx 类名）
        if "::" in nodeid:
            class_part = nodeid.split("::")[1] if "::" in nodeid else ""
            if class_part.startswith("Test"):
                # 提取通路名（去掉 Test 前缀和 Dynamics/Benchmark 后缀）
                pw = class_part[4:]
                for suffix in ("Dynamics", "Benchmark", "Accuracy", "Solver",
                               "Detector", "Conservation", "Approximation"):
                    if pw.endswith(suffix):
                        pw = pw[: -len(suffix)]
                        break
                if pw:
                    by_pathway[pw] += 1

        if outcome == "failed":
            failures.append({
                "nodeid": nodeid,
                "file": file_path,
                "message": r.get("longrepr", "")[:500],
                "duration": r.get("duration", 0.0),
            })
        elif outcome == "skipped":
            skips.append({
                "nodeid": nodeid,
                "file": file_path,
                "reason": r.get("longrepr", "")[:200],
            })

    total = sum(by_outcome.values())
    passed = by_outcome.get("passed", 0)
    failed = by_outcome.get("failed", 0)
    skipped = by_outcome.get("skipped", 0)
    errors = by_outcome.get("error", 0)
    pass_rate = (passed / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "by_outcome": dict(by_outcome),
        "by_file": {k: dict(v) for k, v in by_file.items()},
        "by_marker": dict(by_marker),
        "by_pathway": dict(by_pathway),
        "pass_rate": round(pass_rate, 2),
        "failures": failures,
        "skips": skips,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# JSON 报告
# --------------------------------------------------------------------------- #
def generate_json_report(records: list[dict[str, Any]],
                         stats: dict[str, Any],
                         summary: dict[str, Any],
                         output_path: Path) -> None:
    """生成 JSON 格式报告。"""
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "statistics": stats,
        "test_cases": records,
    }
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Markdown 报告
# --------------------------------------------------------------------------- #
def generate_markdown_report(stats: dict[str, Any],
                             summary: dict[str, Any],
                             output_path: Path) -> None:
    """生成 Markdown 格式报告。"""
    lines: list[str] = []
    lines.append("# BioDynamics v4 验证报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("## 1. 总体统计")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 用例总数 | {stats['total']} |")
    lines.append(f"| 通过 (passed) | {stats['passed']} |")
    lines.append(f"| 失败 (failed) | {stats['failed']} |")
    lines.append(f"| 跳过 (skipped) | {stats['skipped']} |")
    lines.append(f"| 错误 (error) | {stats['errors']} |")
    lines.append(f"| 通过率 | {stats['pass_rate']}% |")
    lines.append("")

    if summary:
        lines.append("## 2. 会话信息")
        lines.append("")
        lines.append("| 字段 | 值 |")
        lines.append("|------|-----|")
        for k, v in summary.items():
            if not isinstance(v, (dict, list)):
                lines.append(f"| {k} | {v} |")
        lines.append("")

    lines.append("## 3. 按测试文件分布")
    lines.append("")
    lines.append("| 文件 | passed | failed | skipped | error | 总计 |")
    lines.append("|------|--------|--------|---------|-------|------|")
    for file_path, counts in sorted(stats["by_file"].items()):
        p = counts.get("passed", 0)
        f = counts.get("failed", 0)
        s = counts.get("skipped", 0)
        e = counts.get("error", 0)
        t = p + f + s + e
        lines.append(f"| {file_path} | {p} | {f} | {s} | {e} | {t} |")
    lines.append("")

    lines.append("## 4. 按 Marker 分布")
    lines.append("")
    if stats["by_marker"]:
        lines.append("| Marker | 用例数 |")
        lines.append("|--------|--------|")
        for m, c in sorted(stats["by_marker"].items(), key=lambda x: -x[1]):
            lines.append(f"| {m} | {c} |")
        lines.append("")
    else:
        lines.append("（无 marker）")
        lines.append("")

    lines.append("## 5. 按通路分布")
    lines.append("")
    if stats["by_pathway"]:
        lines.append("| 通路 | 用例数 |")
        lines.append("|------|--------|")
        for pw, c in sorted(stats["by_pathway"].items(), key=lambda x: -x[1]):
            lines.append(f"| {pw} | {c} |")
        lines.append("")
    else:
        lines.append("（无通路分类）")
        lines.append("")

    if stats["failures"]:
        lines.append("## 6. 失败用例")
        lines.append("")
        for i, fail in enumerate(stats["failures"][:50], 1):
            lines.append(f"### {i}. `{fail['nodeid']}`")
            lines.append("")
            lines.append(f"- **文件**: {fail['file']}")
            lines.append(f"- **耗时**: {fail['duration']:.3f}s")
            lines.append("- **错误信息**:")
            lines.append("```")
            lines.append(fail["message"])
            lines.append("```")
            lines.append("")
        if len(stats["failures"]) > 50:
            lines.append(f"_...其余 {len(stats['failures']) - 50} 条失败见 JSON 报告_")
            lines.append("")

    if stats["skips"]:
        lines.append("## 7. 跳过用例")
        lines.append("")
        lines.append("| 用例 | 原因 |")
        lines.append("|------|------|")
        for skip in stats["skips"][:100]:
            reason = skip["reason"].replace("\n", " ")[:120]
            lines.append(f"| `{skip['nodeid']}` | {escape_md(reason)} |")
        if len(stats["skips"]) > 100:
            lines.append("")
            lines.append(f"_...其余 {len(stats['skips']) - 100} 条跳过见 JSON 报告_")
        lines.append("")

    lines.append("## 8. 验证层级覆盖")
    lines.append("")
    lines.append("| 层级 | 描述 | 状态 |")
    lines.append("|------|------|------|")
    lines.append("| L1 | 单元测试 | 自动 |")
    lines.append("| L2 | 集成测试 | 自动 |")
    lines.append("| L3 | 系统测试 | 自动 |")
    lines.append("| L4 | BioModels 回归 | 自动 |")
    lines.append("| L5 | 文献基准 | 自动 |")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def escape_md(text: str) -> str:
    """转义 Markdown 表格中的特殊字符。"""
    return text.replace("|", "\\|").replace("\n", " ")


# --------------------------------------------------------------------------- #
# HTML 报告
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>BioDynamics v4 验证报告</title>
<style>
  body {{
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    margin: 0;
    padding: 20px;
    background: #f5f7fa;
    color: #2c3e50;
  }}
  h1, h2, h3 {{ color: #1a2a3a; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  .header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    border-radius: 8px;
    margin-bottom: 20px;
  }}
  .header h1 {{ margin: 0 0 10px 0; color: white; }}
  .header .meta {{ font-size: 14px; opacity: 0.9; }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
    margin-bottom: 20px;
  }}
  .stat-card {{
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    text-align: center;
  }}
  .stat-card .value {{ font-size: 32px; font-weight: bold; margin: 8px 0; }}
  .stat-card .label {{ font-size: 13px; color: #7f8c8d; text-transform: uppercase; }}
  .stat-card.passed .value {{ color: #27ae60; }}
  .stat-card.failed .value {{ color: #e74c3c; }}
  .stat-card.skipped .value {{ color: #f39c12; }}
  .stat-card.total .value {{ color: #3498db; }}
  .stat-card.rate .value {{ color: #9b59b6; }}
  .section {{
    background: white;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    font-size: 14px;
  }}
  th, td {{
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid #ecf0f1;
  }}
  th {{
    background: #f8f9fa;
    font-weight: 600;
    color: #2c3e50;
  }}
  tr:hover {{ background: #f8f9fa; }}
  .badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
  }}
  .badge.passed {{ background: #d4edda; color: #155724; }}
  .badge.failed {{ background: #f8d7da; color: #721c24; }}
  .badge.skipped {{ background: #fff3cd; color: #856404; }}
  .badge.error {{ background: #f8d7da; color: #721c24; }}
  pre {{
    background: #f8f9fa;
    padding: 10px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 12px;
    border-left: 3px solid #e74c3c;
  }}
  .failure-item {{
    border-left: 3px solid #e74c3c;
    padding: 10px 15px;
    margin: 10px 0;
    background: #fdf8f8;
    border-radius: 4px;
  }}
  .failure-item .nodeid {{
    font-family: monospace;
    color: #c0392b;
    font-weight: 600;
    word-break: break-all;
  }}
  .progress-bar {{
    height: 24px;
    background: #ecf0f1;
    border-radius: 12px;
    overflow: hidden;
    margin: 10px 0;
  }}
  .progress-bar .fill {{
    height: 100%;
    background: linear-gradient(90deg, #27ae60, #2ecc71);
    transition: width 0.5s;
  }}
  .footer {{
    text-align: center;
    color: #7f8c8d;
    font-size: 12px;
    margin-top: 20px;
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>BioDynamics v4 验证报告</h1>
    <div class="meta">生成时间：{generated_at}</div>
  </div>

  <div class="stats-grid">
    <div class="stat-card total">
      <div class="label">用例总数</div>
      <div class="value">{total}</div>
    </div>
    <div class="stat-card passed">
      <div class="label">通过</div>
      <div class="value">{passed}</div>
    </div>
    <div class="stat-card failed">
      <div class="label">失败</div>
      <div class="value">{failed}</div>
    </div>
    <div class="stat-card skipped">
      <div class="label">跳过</div>
      <div class="value">{skipped}</div>
    </div>
    <div class="stat-card rate">
      <div class="label">通过率</div>
      <div class="value">{pass_rate}%</div>
    </div>
  </div>

  <div class="progress-bar">
    <div class="fill" style="width: {pass_rate}%;"></div>
  </div>

  <div class="section">
    <h2>按测试文件分布</h2>
    <table>
      <thead>
        <tr><th>文件</th><th>通过</th><th>失败</th><th>跳过</th><th>错误</th><th>总计</th></tr>
      </thead>
      <tbody>
{file_rows}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>按通路分布</h2>
    <table>
      <thead>
        <tr><th>通路</th><th>用例数</th></tr>
      </thead>
      <tbody>
{pathway_rows}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>按 Marker 分布</h2>
    <table>
      <thead>
        <tr><th>Marker</th><th>用例数</th></tr>
      </thead>
      <tbody>
{marker_rows}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>失败用例 ({failure_count})</h2>
{failure_section}
  </div>

  <div class="section">
    <h2>跳过用例 ({skip_count})</h2>
    <table>
      <thead>
        <tr><th>用例</th><th>原因</th></tr>
      </thead>
      <tbody>
{skip_rows}
      </tbody>
    </table>
  </div>

  <div class="footer">
    BioDynamics v4 Verification Suite · 自动生成 · Powered by pytest
  </div>
</div>
</body>
</html>
"""


def generate_html_report(stats: dict[str, Any],
                         output_path: Path) -> None:
    """生成 HTML 格式报告。"""
    # 文件分布行
    file_rows: list[str] = []
    for file_path, counts in sorted(stats["by_file"].items()):
        p = counts.get("passed", 0)
        f = counts.get("failed", 0)
        s = counts.get("skipped", 0)
        e = counts.get("error", 0)
        t = p + f + s + e
        file_rows.append(
            f'        <tr><td>{escape(file_path)}</td><td>{p}</td>'
            f'<td>{f}</td><td>{s}</td><td>{e}</td><td>{t}</td></tr>'
        )
    file_rows_str = "\n".join(file_rows) if file_rows else '        <tr><td colspan="6">无数据</td></tr>'

    # 通路分布行
    pathway_rows: list[str] = []
    for pw, c in sorted(stats["by_pathway"].items(), key=lambda x: -x[1]):
        pathway_rows.append(
            f'        <tr><td>{escape(pw)}</td><td>{c}</td></tr>'
        )
    pathway_rows_str = "\n".join(pathway_rows) if pathway_rows else '        <tr><td colspan="2">无通路分类</td></tr>'

    # Marker 分布行
    marker_rows: list[str] = []
    for m, c in sorted(stats["by_marker"].items(), key=lambda x: -x[1]):
        marker_rows.append(
            f'        <tr><td>{escape(m)}</td><td>{c}</td></tr>'
        )
    marker_rows_str = "\n".join(marker_rows) if marker_rows else '        <tr><td colspan="2">无 marker</td></tr>'

    # 失败用例区块
    failure_section: list[str] = []
    for fail in stats["failures"][:30]:
        failure_section.append('<div class="failure-item">')
        failure_section.append(f'<div class="nodeid">{escape(fail["nodeid"])}</div>')
        failure_section.append(f'<div>文件: {escape(fail["file"])}</div>')
        failure_section.append(f'<div>耗时: {fail["duration"]:.3f}s</div>')
        failure_section.append('<pre>')
        failure_section.append(escape(fail["message"]))
        failure_section.append('</pre>')
        failure_section.append('</div>')
    if len(stats["failures"]) > 30:
        failure_section.append(
            f'<p><em>...其余 {len(stats["failures"]) - 30} 条失败见 JSON 报告</em></p>'
        )
    if not stats["failures"]:
        failure_section.append('<p>无失败用例。</p>')
    failure_section_str = "\n".join(failure_section)

    # 跳过用例行
    skip_rows: list[str] = []
    for skip in stats["skips"][:100]:
        reason = skip["reason"].replace("\n", " ")[:150]
        skip_rows.append(
            f'        <tr><td>{escape(skip["nodeid"])}</td>'
            f'<td>{escape(reason)}</td></tr>'
        )
    if len(stats["skips"]) > 100:
        skip_rows.append(
            f'        <tr><td colspan="2"><em>...其余 {len(stats["skips"]) - 100} 条见 JSON 报告</em></td></tr>'
        )
    if not stats["skips"]:
        skip_rows.append('        <tr><td colspan="2">无跳过用例。</td></tr>')
    skip_rows_str = "\n".join(skip_rows)

    html = HTML_TEMPLATE.format(
        generated_at=datetime.now().isoformat(),
        total=stats["total"],
        passed=stats["passed"],
        failed=stats["failed"],
        skipped=stats["skipped"],
        pass_rate=stats["pass_rate"],
        file_rows=file_rows_str,
        pathway_rows=pathway_rows_str,
        marker_rows=marker_rows_str,
        failure_count=len(stats["failures"]),
        failure_section=failure_section_str,
        skip_count=len(stats["skips"]),
        skip_rows=skip_rows_str,
    )
    output_path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="BioDynamics v4 验证报告生成器"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("reports"),
        help="pytest 结果目录（含 _collected.jsonl 和 summary.json）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="报告输出目录",
    )
    args = parser.parse_args(argv)

    results_dir: Path = args.results_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    records = load_collected(results_dir)
    summary = load_summary(results_dir)

    if not records and not summary:
        print(f"[警告] 在 {results_dir} 未找到 _collected.jsonl 或 summary.json", file=sys.stderr)
        print("       请先运行 pytest 触发 conftest.py 的钩子写入数据。", file=sys.stderr)

    # 聚合统计
    stats = aggregate_statistics(records)

    # 生成三种格式
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    html_path = output_dir / "report.html"

    generate_json_report(records, stats, summary, json_path)
    generate_markdown_report(stats, summary, md_path)
    generate_html_report(stats, html_path)

    print(f"[OK] 已生成报告：")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    print(f"  - {html_path}")
    print(f"\n统计摘要：")
    print(f"  用例总数: {stats['total']}")
    print(f"  通过: {stats['passed']}  失败: {stats['failed']}  "
          f"跳过: {stats['skipped']}  错误: {stats['errors']}")
    print(f"  通过率: {stats['pass_rate']}%")

    return 0 if stats["failed"] == 0 and stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
