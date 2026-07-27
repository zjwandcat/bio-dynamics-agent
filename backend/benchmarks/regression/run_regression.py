"""
Scientific Regression Check — Sprint 0 Task 0.3
目的：检测科学指标退化，阻断导致科学正确性下降的 PR
原则：所有结论基于真实运行结果，不接受 LLM 自述
"""
import json
import sys
from pathlib import Path

# 回归检查规则定义
# 每条规则：指标名 + 阈值 + 比较方式 + 报警消息
REGRESSION_RULES = [
    {
        "name": "erk_peak_time",
        "metric": "simulation.erk_peak_min",
        "baseline": 20.0,
        "max_deviation_pct": 0.5,  # 允许 50% 偏移（10-20 min 范围）
        "comparison": "max",
        "alarm": "ERK Peak 时间退化：基线 20 min，实际超过允许范围（可能变为 40 min）"
    },
    {
        "name": "egfr_peak_time",
        "metric": "simulation.egfr_peak_min",
        "baseline": 5.0,
        "max_deviation_pct": 1.0,
        "comparison": "max",
        "alarm": "EGFR Peak 时间退化：基线 <5 min，实际超过 10 min"
    },
    {
        "name": "discussion_topic_drift",
        "metric": "discussion.topic_keywords",
        "forbidden_keywords": ["癌症耐药", "cancer drug resistance", "EGFR qPCR"],
        "comparison": "forbidden_contains",
        "alarm": "Discussion 主题漂移：出现禁用关键词（如癌症耐药/qPCR）"
    },
    {
        "name": "pmid_hallucination",
        "metric": "discussion.cited_pmids",
        "known_hallucination_pmids": ["18050474", "39059397", "40333694"],
        "comparison": "not_contains",
        "alarm": "PMID 幻觉复发：报告引用了审计已确认的幻觉 PMID"
    },
    {
        "name": "confidence_inflation",
        "metric": "validation.confidence",
        "max_acceptable": 0.95,
        "comparison": "max_acceptable",
        "alarm": "Confidence 虚高：超过 0.95 可能未真实执行 Consistency 检查"
    }
]


def load_baseline():
    """加载基线指标（待 Sprint 1 填充真实基线）"""
    baseline_path = Path(__file__).parent / "baseline.json"
    if baseline_path.exists():
        # utf-8-sig 兼容带 BOM 和不带 BOM 的 UTF-8
        return json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    # 无基线时返回空（首次运行不阻断，仅记录）
    return {}


def load_actual():
    """加载实际运行指标（待各 Sprint 接通真实运行结果）"""
    actual_path = Path(__file__).parent / "actual.json"
    if actual_path.exists():
        # utf-8-sig 兼容带 BOM 和不带 BOM 的 UTF-8
        return json.loads(actual_path.read_text(encoding="utf-8-sig"))
    return None  # 无实际结果时跳过（不阻断）


def check_rule(rule, actual):
    """检查单条规则，返回 (passed, message)"""
    name = rule["name"]
    comparison = rule.get("comparison", "")

    if comparison == "max":
        # 实际值不得超过 baseline * (1 + max_deviation_pct)
        metric_path = rule["metric"]
        value = _get_nested(actual, metric_path)
        if value is None:
            return True, f"{name}: skipped (no actual value)"
        baseline = rule["baseline"]
        threshold = baseline * (1 + rule["max_deviation_pct"])
        if value > threshold:
            return False, f"{name}: FAIL — actual={value}, threshold={threshold}. {rule['alarm']}"
        return True, f"{name}: PASS — actual={value}, threshold={threshold}"

    if comparison == "max_acceptable":
        # 实际值不得超过绝对上限 max_acceptable（无 baseline，用于 confidence 等绝对阈值）
        metric_path = rule["metric"]
        value = _get_nested(actual, metric_path)
        if value is None:
            return True, f"{name}: skipped (no actual value)"
        threshold = rule["max_acceptable"]
        if value > threshold:
            return False, f"{name}: FAIL — actual={value}, max_acceptable={threshold}. {rule['alarm']}"
        return True, f"{name}: PASS — actual={value}, max_acceptable={threshold}"

    if comparison == "forbidden_contains":
        # 实际值不得包含禁用关键词
        metric_path = rule["metric"]
        value = _get_nested(actual, metric_path)
        if value is None:
            return True, f"{name}: skipped (no actual value)"
        forbidden = rule.get("forbidden_keywords", [])
        violations = [kw for kw in forbidden if kw in str(value)]
        if violations:
            return False, f"{name}: FAIL — forbidden keywords {violations}. {rule['alarm']}"
        return True, f"{name}: PASS — no forbidden keywords"

    if comparison == "not_contains":
        # 实际值不得包含已知幻觉 PMID
        metric_path = rule["metric"]
        value = _get_nested(actual, metric_path)
        if value is None:
            return True, f"{name}: skipped (no actual value)"
        hallucination_pmids = rule.get("known_hallucination_pmids", [])
        actual_pmids = [str(p) for p in (value if isinstance(value, list) else [value])]
        violations = [p for p in hallucination_pmids if p in actual_pmids]
        if violations:
            return False, f"{name}: FAIL — hallucination PMIDs {violations}. {rule['alarm']}"
        return True, f"{name}: PASS — no hallucination PMIDs"

    return True, f"{name}: unknown comparison {comparison}"


def _get_nested(data, path):
    """按 a.b.c 路径获取嵌套值"""
    if data is None:
        return None
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def main():
    """主入口：运行全部回归规则，生成报告，决定是否阻断"""
    print("=" * 60)
    print("Scientific Regression Check — Sprint 0 Task 0.3")
    print("=" * 60)

    actual = load_actual()
    if actual is None:
        print("[INFO] 无 actual.json，跳过回归检查（首次运行或无运行结果）")
        # 生成空报告
        report = {
            "status": "SKIPPED",
            "reason": "no actual.json found",
            "rules_checked": 0,
            "rules_passed": 0,
            "rules_failed": 0,
            "failures": []
        }
    else:
        results = []
        failures = []
        for rule in REGRESSION_RULES:
            passed, message = check_rule(rule, actual)
            results.append({"name": rule["name"], "passed": passed, "message": message})
            if not passed:
                failures.append(rule["name"])

        report = {
            "status": "PASS" if not failures else "FAIL",
            "rules_checked": len(results),
            "rules_passed": len([r for r in results if r["passed"]]),
            "rules_failed": len(failures),
            "failures": failures,
            "details": results
        }

    # 写报告
    report_path = Path(__file__).parent / "regression_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 报告已生成: {report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 阻断逻辑：FAIL 时退出码 1
    if report["status"] == "FAIL":
        print(f"[BLOCK] Scientific Regression FAIL — 阻断 Merge")
        sys.exit(1)
    else:
        print(f"[OK] Scientific Regression {report['status']}")
        sys.exit(0)


if __name__ == "__main__":
    main()
