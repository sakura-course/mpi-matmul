.PHONY: list run run-benchmark

list:
	@echo "Available targets:"
	@echo "  list          - List all available targets"
	@echo "  run           - Run the main program"
	@echo "  run-benchmark - Run the benchmark"

run:
	mpiexec -n 4 uv run python src/main.py

run-benchmark:
	uv run python src/benchmark.py
