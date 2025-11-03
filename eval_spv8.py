import os
import subprocess

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
EXEC_DIR = os.path.join(BASE_PATH, "spv8-public", "data")

def extract_timing(output_text):
    """Extract timing information from the program output."""
    for line in output_text.split('\n'):
        if "Time:" in line:
            # Extract the time value
            time_str = line.split(":")[-1].strip().split()[0]
            return float(time_str)

def write_per_matrix_csvs():
    results_file_path = os.path.join(BASE_PATH, "spv8_eval_results.txt")
    output_dir = os.path.join(BASE_PATH, "results")

    # Read and collect timings grouped by base matrix name
    grouped = {}
    if not os.path.exists(results_file_path):
        print(f"Results file not found: {results_file_path}. Skipping CSV creation.")
        return

    with open(results_file_path, "r") as fr:
        for line in fr:
            line = line.strip()
            if not line or line.startswith("Matrix"):
                continue
            # Expect lines like: name,time
            parts = line.split(",", 1)
            if len(parts) != 2:
                raise Exception(f"Malformed line in results file: {line}")
            name, time_str = parts[0].strip(), parts[1].strip()
            time_val = float(time_str)

            # Interpret names like `{matrix}_random_{num}pct` or bare `{matrix}`
            if "_random_" in name:
                base, tail = name.split("_random_", 1)
                # tail expected like '90pct'
                if tail.endswith("pct"):
                    pct_str = tail[:-3]
                    try:
                        pct = int(pct_str)
                    except ValueError:
                        # fallback: mark as unknown (skip)
                        continue
                else:
                    # no pct suffix: skip
                    continue
            else:
                base = name
                pct = 100

            grouped.setdefault(base, []).append((pct, time_val))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write per-matrix CSV files
    for base, entries in grouped.items():
        # sort by percentage descending so 100 comes first, then 90, etc.
        entries_sorted = sorted(entries, key=lambda x: x[0], reverse=True)
        out_name = f"timing_{base}_SpMV_random.csv"
        out_path = os.path.join(output_dir, out_name)
        try:
            with open(out_path, "w") as fo:
                fo.write("Percentage,time\n")
                for pct, t in entries_sorted:
                    fo.write(f"{pct},{t:.6f}\n")
            print(f"Wrote {out_path} ({len(entries_sorted)} rows)")
        except OSError as e:
            print(f"Failed to write {out_path}: {e}")

if __name__ == "__main__":
    with open("spv8_eval_results.txt", "w") as fresults:
        fresults.write("Matrix,Time\n")
        for dir_name in os.listdir(EXEC_DIR):
            # Check if it's a directory
            dir_path = os.path.join(EXEC_DIR, dir_name)
            if not os.path.isdir(dir_path):
                continue
            cmd = ["../../bin/spmv_spv8", "100", "0", "1"]
            result = subprocess.run(cmd, cwd=os.path.join(EXEC_DIR, dir_name), capture_output=True, text=True, check=True)
            
            timing_info = extract_timing(result.stdout)
            print("\n" + "=" * 60)
            print("Matrix:", dir_name)
            print("=" * 60)
            print(f"{timing_info:.6f}")
            print("=" * 60)
            fresults.write(f"{dir_name},{timing_info:.6f}\n")
            fresults.flush()
    write_per_matrix_csvs()


