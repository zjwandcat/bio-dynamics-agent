"""Sprint 1 验证脚本：全 10 通路 YAML/JSON 格式 + 字段完整性 + BioModels ID 一致性检查。

验证项：
1. 所有 canonical/*.yaml 能解析且含 canonical_timeline
2. 所有 benchmark.yaml 能解析且含 10 个必需字段
3. 所有 expected_metrics.json 能解析且含 8 项 metrics
4. BioModels ID 一致性：benchmark.yaml.canonical_biomodels == expected_metrics.json.biomodels.expected
5. canonical/*.yaml.canonical_models 与 benchmark.yaml.canonical_biomodels 一致
"""
import json
import sys
from pathlib import Path

import yaml

BACKEND = Path(__file__).resolve().parent.parent
CANONICAL_DIR = BACKEND / "knowledge" / "canonical"
GOLDEN_DIR = BACKEND / "benchmarks" / "golden"

PATHWAYS = ["EGFR", "MAPK", "PI3K", "p53", "NFKB", "JAKSTAT", "TGFB", "WNT", "Apoptosis", "CellCycle"]

# canonical YAML 文件名映射（部分使用下划线小写）
CANONICAL_FILE_MAP = {
    "EGFR": "egfr.yaml",
    "MAPK": "mapk.yaml",
    "PI3K": "pi3k_akt_mtor.yaml",
    "p53": "p53.yaml",
    "NFKB": "nf_kappa_b.yaml",
    "JAKSTAT": "jak_stat.yaml",
    "TGFB": "tgf_beta.yaml",
    "WNT": "wnt.yaml",
    "Apoptosis": "apoptosis.yaml",
    "CellCycle": "cell_cycle.yaml",
}

BENCHMARK_REQUIRED_FIELDS = [
    "pathway", "question", "ground_truth", "canonical_papers",
    "canonical_biomodels", "expected_mechanism", "expected_time_scale",
    "expected_experiment", "expected_discussion", "expected_validation",
    "expected_confidence",
]

METRICS_KEYS = [
    "literature", "biomodels", "mechanism", "simulation",
    "validation", "discussion", "experiment", "scientific_consistency",
]


def main() -> int:
    errors = []
    warnings = []

    for pathway in PATHWAYS:
        # 1. canonical YAML
        canon_file = CANONICAL_DIR / CANONICAL_FILE_MAP[pathway]
        if not canon_file.exists():
            errors.append(f"[{pathway}] canonical 文件缺失: {canon_file.name}")
            continue
        try:
            canon = yaml.safe_load(canon_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            errors.append(f"[{pathway}] canonical YAML 解析失败: {e}")
            continue
        if "canonical_timeline" not in canon:
            errors.append(f"[{pathway}] canonical 缺少 canonical_timeline 字段")
        canon_models = canon.get("canonical_models", [])

        # 2. benchmark.yaml
        bench_file = GOLDEN_DIR / pathway / "benchmark.yaml"
        if not bench_file.exists():
            errors.append(f"[{pathway}] benchmark.yaml 缺失")
            continue
        try:
            bench = yaml.safe_load(bench_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            errors.append(f"[{pathway}] benchmark.yaml 解析失败: {e}")
            continue
        missing_fields = [f for f in BENCHMARK_REQUIRED_FIELDS if f not in bench]
        if missing_fields:
            errors.append(f"[{pathway}] benchmark.yaml 缺字段: {missing_fields}")
        bench_biomodels = bench.get("canonical_biomodels", [])

        # 3. expected_metrics.json
        metrics_file = GOLDEN_DIR / pathway / "expected_metrics.json"
        if not metrics_file.exists():
            errors.append(f"[{pathway}] expected_metrics.json 缺失")
            continue
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errors.append(f"[{pathway}] expected_metrics.json 解析失败: {e}")
            continue
        metrics_keys = list(metrics.get("metrics", {}).keys())
        missing_metrics = [k for k in METRICS_KEYS if k not in metrics_keys]
        if missing_metrics:
            errors.append(f"[{pathway}] expected_metrics.json 缺 metric: {missing_metrics}")
        metrics_biomodels_expected = metrics.get("metrics", {}).get("biomodels", {}).get("expected", "MISSING")

        # 4. BioModels ID 一致性：benchmark vs expected_metrics
        if sorted(bench_biomodels) != sorted(metrics_biomodels_expected):
            errors.append(
                f"[{pathway}] BioModels 不一致: benchmark.yaml={bench_biomodels} "
                f"vs expected_metrics.json={metrics_biomodels_expected}"
            )

        # 5. canonical_models vs benchmark.canonical_biomodels 一致性
        if sorted(canon_models) != sorted(bench_biomodels):
            warnings.append(
                f"[{pathway}] canonical_models({canon_models}) != "
                f"benchmark.canonical_biomodels({bench_biomodels})"
            )

    # 输出结果
    print("=" * 70)
    print("Sprint 1 验证报告")
    print("=" * 70)
    print(f"通路数: {len(PATHWAYS)}")
    print(f"错误数: {len(errors)}")
    print(f"警告数: {len(warnings)}")
    print()

    if errors:
        print("=== ERRORS (必须修复) ===")
        for e in errors:
            print(f"  [ERROR] {e}")
        print()

    if warnings:
        print("=== WARNINGS (建议检查) ===")
        for w in warnings:
            print(f"  [WARN]  {w}")
        print()

    if not errors:
        print("[PASS] 所有 10 通路 Sprint 1 数据完整性验证通过")
        return 0
    else:
        print("[FAIL] 存在错误，请修复后再进 Sprint 1 回归测试")
        return 1


if __name__ == "__main__":
    sys.exit(main())
