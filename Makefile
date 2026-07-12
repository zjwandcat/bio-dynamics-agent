.PHONY: test test-unit test-integration test-benchmark test-coverage ci benchmark benchmark-legacy

test:
	cd backend && python -m pytest tests/ -v --tb=short

test-unit:
	cd backend && python -m pytest tests/ -v --tb=short -m "not slow"

test-integration:
	cd backend && python -m pytest tests/ -k "integration or e2e" -v

test-benchmark:
	cd verification && python -m pytest benchmark/ -v --tb=short

test-coverage:
	cd backend && python -m pytest tests/ --cov=app --cov-report=html

# Task 18 SubTask 18.2: make benchmark
# 调用真实端到端编排器跑完全部 10 通路，输出 benchmark_results.md
# 需要 .env 中配置 LLM API key 等环境变量
benchmark:
	cd backend && BENCHMARK_REAL_ORCHESTRATOR=true python -c "from app.benchmark_runner import BenchmarkRunner; r = BenchmarkRunner(); s = r.run_all_to_markdown('benchmark_results.md'); print(f'Total={s[\"total\"]} Passed={s[\"passed\"]} Failed={s[\"failed\"]}')" 2>&1 | tee benchmark_output.log

# Task 18: legacy synthetic benchmark（已废弃，仅快速 schema 检查时使用）
benchmark-legacy:
	cd backend && BENCHMARK_LEGACY_SYNTHETIC=true python -c "from app.benchmark_runner import BenchmarkRunner; r = BenchmarkRunner(); s = r.run_all_to_markdown('benchmark_results.md'); print(f'Total={s[\"total\"]} Passed={s[\"passed\"]} Failed={s[\"failed\"]}')" 2>&1 | tee benchmark_output.log

ci: test test-integration test-benchmark
	@echo "CI pipeline complete"
