import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_matrix_timing(matrix, op, reduction_type):
    df = pd.read_csv(f"results/papi_{matrix}_{op}_{reduction_type}.csv")
    plt.figure(figsize=(8,6))
    plt.plot(df["Percentage"], df["branch_mispreds"], marker="o", linestyle="-")

    plt.title(f"Matrix Density vs Branch Mispredictions ({matrix} - {reduction_type})")
    plt.xlabel("Density (%)")
    plt.ylabel("Branch Mispredictions")
    plt.gca().invert_xaxis()  # so 100% appears on the left
    plt.grid(True)

    # Show the plot
    plt.savefig(f"plots/{matrix}_{op}_{reduction_type}.pdf")
    plt.close()


if __name__ == "__main__":
    matrices = [p.stem for p in Path("matrices").glob("*.mtx")]
    # ops = ["spmm", "spmv"]
    ops = ["spmv"]
    reduction_types = ["random", "truncated", "consec"]
    for matrix in matrices:
        for op in ops:
            for reduction_type in reduction_types:
                plot_matrix_timing(matrix, op, reduction_type)
