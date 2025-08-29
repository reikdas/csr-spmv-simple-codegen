import pandas as pd
import matplotlib.pyplot as plt

def plot_matrix_timing(matrix, op):
    df = pd.read_csv(f"timing_{matrix}_{op}.csv")
    plt.figure(figsize=(8,6))
    plt.plot(df["Percentage"], df["Time_ns"], marker="o", linestyle="-")

    plt.title(f"Matrix Density vs Time ({matrix})")
    plt.xlabel("Density (%)")
    plt.ylabel("Time (ns)")
    plt.gca().invert_xaxis()  # so 100% appears on the left
    plt.grid(True)

    # Show the plot
    plt.savefig(f"{matrix}_{op}.pdf")


if __name__ == "__main__":
    plot_matrix_timing("brainpc2", "spmv")
