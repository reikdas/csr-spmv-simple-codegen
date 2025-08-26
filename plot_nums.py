import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("timing_results.csv")  # replace with your filename

# Plot
plt.figure(figsize=(8,6))
plt.plot(df["Percentage"], df["Time_ns"], marker="o", linestyle="-")

plt.title("Matrix Density vs Time")
plt.xlabel("Density (%)")
plt.ylabel("Time (ns)")
plt.gca().invert_xaxis()  # so 100% appears on the left
plt.grid(True)

# Show the plot
plt.savefig("matrix_density_vs_time.pdf")
