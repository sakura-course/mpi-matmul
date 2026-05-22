"""
MPI并行矩阵乘法 - 按行划分 (Row-wise Decomposition)

计算 C = A × B
A: m×k, B: k×n, C: m×n

运行: mpiexec -n 4 uv run src/main.py 512 512 512
"""

import sys

import numpy as np
from mpi4py import MPI


def main():
    """
    MPI并行矩阵乘法主函数

    从命令行读取矩阵规模(m, k, n)和计算模式，按行划分A矩阵到各进程，
    广播B矩阵，各进程计算本地子块后收集到rank 0输出耗时和GFLOPS。

    Args:
        (通过sys.argv传入) m, k, n, mode
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # 解析命令行参数
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    mode = sys.argv[4] if len(sys.argv) > 4 else "numpy"
    assert mode in ("numpy", "loop"), "模式必须是 'numpy' 或 'loop'"

    if rank == 0:
        print("=== MPI 并行矩阵乘法 (按行划分) ===")
        print(f"矩阵规模: A({m}×{k}) × B({k}×{n}) = C({m}×{n})")
        print(f"进程数: {nprocs}")
        print(f"计算模式: {mode}")
        print("-----------------------------------")

    # 按行划分: 计算每个进程分到的行数
    counts = [m // nprocs + (1 if i < m % nprocs else 0) for i in range(nprocs)]
    offsets = [sum(counts[:i]) for i in range(nprocs)]
    local_rows = counts[rank]

    # 初始化矩阵
    if rank == 0:
        np.random.seed(42)
        A = np.random.rand(m, k).astype(np.float64)
        np.random.seed(123)
        B = np.random.rand(k, n).astype(np.float64)
    else:
        A = None
        B = np.empty((k, n), dtype=np.float64)

    t_start = MPI.Wtime()

    # 广播B
    comm.Bcast(B, root=0)

    # 按行分发A
    local_A = np.empty((local_rows, k), dtype=np.float64)
    s_counts = [c * k for c in counts]
    s_offsets = [o * k for o in offsets]
    comm.Scatterv([A, s_counts, s_offsets, MPI.DOUBLE], local_A, root=0)

    # 同步
    comm.Barrier()

    if mode == "loop":
        local_C = np.empty((local_rows, n), dtype=np.float64)
        for i in range(local_rows):
            for j in range(n):
                s = 0.0
                for p in range(k):
                    s += local_A[i, p] * B[p, j]
                local_C[i, j] = s
    else:
        local_C = local_A @ B

    comm.Barrier()

    # 收集结果
    C = np.empty((m, n), dtype=np.float64) if rank == 0 else None
    g_counts = [c * n for c in counts]
    g_offsets = [o * n for o in offsets]
    comm.Gatherv(local_C, [C, g_counts, g_offsets, MPI.DOUBLE], root=0)

    t_end = MPI.Wtime()
    t_parallel = t_end - t_start

    # 输出结果
    if rank == 0:
        print(f"并行计算耗时: {t_parallel:.6f} 秒")
        gflops = 2.0 * m * k * n / t_parallel / 1e9
        print(f"GFLOPS: {gflops:.3f}")


if __name__ == "__main__":
    main()
