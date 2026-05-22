# mpi-matmul

利用 MPI 实现矩阵乘法的并行计算，采用按行划分（Row-wise Decomposition）策略。

## 前置要求

* `uv`
* `msmpi`
* `make`

## 用法

```bash
# Run the main program
make run
# or
mpiexec -n 4 uv run python src/main.py

# Run the benchmark
make run-benchmark
# or
uv run python src/benchmark.py
```
