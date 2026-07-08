"""Release Gate：定义发布门禁条件。"""
# 所有检查必须通过才能发布
RELEASE_GATE_CHECKS = [
    "unit_tests",
    "integration_tests",
    "scientific_benchmarks",
    "biomodels_regression",
    "parameter_stress",
    "playwright_ui",
    "performance_benchmark",
    "coverage_report",
]
