.PHONY: test test-unit test-integration test-benchmark test-coverage ci

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

ci: test test-integration test-benchmark
	@echo "CI pipeline complete"
