import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_matrix_timing(matrix, op, reduction_type):
    df = pd.read_csv(f"results/timing_{matrix}_{op}_{reduction_type}.csv")
    plt.figure(figsize=(8,6))
    plt.plot(df["Percentage"], df["Time_ns"], marker="o", linestyle="-")

    plt.title(f"Matrix Density vs Time ({matrix} - {reduction_type})")
    plt.xlabel("Density (%)")
    plt.ylabel("Time (ns)")
    plt.gca().invert_xaxis()  # so 100% appears on the left
    plt.grid(True)

    # Show the plot
    plt.savefig(f"plots/{matrix}_{op}_{reduction_type}.pdf")


if __name__ == "__main__":
    matrices = [p.stem for p in Path("matrices").glob("*.mtx")]
    ops = ["spmm", "spmv"]
    reduction_types = ["random", "truncated", "consec"]
    for matrix in matrices:
        for op in ops:
            for reduction_type in reduction_types:
                plot_matrix_timing(matrix, op, reduction_type)
