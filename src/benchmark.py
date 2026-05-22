"""
批量性能测试脚本

用法: python src/benchmark.py
"""

import subprocess
import re
import os


# 测试的矩阵规模列表
SIZES = [256, 512, 1024]
# 测试的进程数列表
NPS = [1, 2, 4, 8]
# 测试的计算模式列表
MODES = ["numpy", "loop"]

os.makedirs("results", exist_ok=True)

with open("results/benchmark.csv", "w") as f:
    f.write("进程数,矩阵大小,numpy耗时(秒),loop耗时(秒)\n")

    for size in SIZES:
        for np in NPS:
            times = {}
            for mode in MODES:
                print(f"测试: np={np}, size={size}x{size}, mode={mode}")
                cmd = f"mpiexec -n {np} uv run src/main.py {size} {size} {size} {mode}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                output = result.stdout

                match = re.search(r"并行计算耗时:\s+([\d.]+)", output)
                if match:
                    t = match.group(1)
                    times[mode] = t
                    print(f"  耗时: {t}s")
                else:
                    print(f"  失败: {output}")

            f.write(
                f"{np},{size},{times.get('numpy', 'N/A')},{times.get('loop', 'N/A')}\n"
            )

print("\n结果已保存到 results/benchmark.csv")
