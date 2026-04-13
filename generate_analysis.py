"""
Analysis: Branch Misprediction as the Root Cause of SpMV Timing Anomalies
under Random NNZ Removal

Experiment recap:
  - We sweep a sparse matrix's NNZ percentage from 100% down to 10%.
  - "random"     removal: remove NNZ uniformly at random across the whole matrix.
  - "consec"     removal: remove NNZ consecutively (from the start of each row).
  - "truncated"  removal: similar consecutive-style removal.

Observation: random removal often makes SpMV *slower* than the full matrix.
Hypothesis: random removal creates irregular row-length distributions that
            defeat the branch predictor inside the CSR inner loop.
"""

import os, csv, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

OUT = "analysis_plots"
os.makedirs(OUT, exist_ok=True)

PERCENTAGES = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

# ── helpers ────────────────────────────────────────────────────────────────

def load_csv(path):
    """Return {percentage: {col: value}} dict."""
    d = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pct = float(row["Percentage"])
            d[pct] = {k: float(v) for k, v in row.items()}
    return d

def sorted_series(data, col):
    """Return (pcts, values) sorted by percentage descending."""
    pcts = sorted(data.keys(), reverse=True)
    return pcts, [data[p][col] for p in pcts]

def get_matrix_name(fn, prefix, suffix):
    return fn[len(prefix):-len(suffix)]

# ── data loading ───────────────────────────────────────────────────────────

def load_sweep(directory, kind="timing"):
    """Load all timing or papi files from a sweep directory."""
    prefix = f"{kind}_"
    suffix = "_SpMV_random.csv"
    result = {}
    for fn in os.listdir(directory):
        if fn.startswith(prefix) and fn.endswith(suffix):
            matrix = get_matrix_name(fn, prefix, suffix)
            result[matrix] = load_csv(os.path.join(directory, fn))
    return result

def load_papi_modes(directory):
    """Load consec/random/truncated papi files; return {matrix: {mode: data}}."""
    result = defaultdict(dict)
    for fn in os.listdir(directory):
        if not fn.startswith("papi_"):
            continue
        # papi_<matrix>_spmv_<mode>.csv  or  papi_<matrix>_SpMV_<mode>.csv
        for sep in ("_spmv_", "_SpMV_"):
            if sep in fn:
                parts = fn[len("papi_"):].split(sep)
                matrix = parts[0]
                mode = parts[1].replace(".csv", "")
                result[matrix][mode] = load_csv(os.path.join(directory, fn))
                break
    return dict(result)

# ── compute summary stats ──────────────────────────────────────────────────

def slowdown_stats(timing_dict):
    """Return list of {matrix, slowdown, peak_pct} dicts."""
    out = []
    for matrix, data in timing_dict.items():
        t100 = data.get(100, {}).get("time")
        if t100 is None or t100 == 0:
            continue
        max_t = max(r["time"] for r in data.values())
        peak_pct = max(data.keys(), key=lambda p: data[p]["time"])
        out.append({"matrix": matrix, "slowdown": max_t / t100, "peak_pct": peak_pct, "t100": t100})
    return out

def mispred_stats(papi_dict, mode="random"):
    """Return {matrix: max_mispred_%} for given mode."""
    out = {}
    for matrix, modes in papi_dict.items():
        if mode not in modes:
            continue
        out[matrix] = max(r["branch_mispreds"] for r in modes[mode].values())
    return out

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 – Timing anomaly examples (6 matrices)
# ══════════════════════════════════════════════════════════════════════════

def fig_timing_examples():
    timing = load_sweep("sweep_matrices_warmup", "timing")
    papi   = load_sweep("sweep_matrices_warmup", "papi")

    # Pick 6 matrices spanning different degrees of anomaly
    targets = ["epb1", "mark3jac040", "bloweybl", "TSOPF_FS_b9_c6",
               "G27", "G39"]
    # G27 and G39 come from the 34-matrix deep-dive set; normalise names
    # (sweep_matrices_warmup uses the same names)
    available = [m for m in targets if m in timing]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    colors = {"time": "#1f77b4", "branch_mispreds": "#d62728"}

    for ax, matrix in zip(axes, available):
        pcts_t, times = sorted_series(timing[matrix], "time")
        t100 = times[0]
        norm_times = [t / t100 for t in times]

        ax2 = ax.twinx()
        if matrix in papi:
            pcts_p, mispreds = sorted_series(papi[matrix], "branch_mispreds")
            ax2.plot(pcts_p, mispreds, color=colors["branch_mispreds"],
                     linestyle="--", marker="s", markersize=4, linewidth=1.3,
                     label="Branch mispred %", zorder=3)
            ax2.set_ylabel("Branch mispred %", color=colors["branch_mispreds"], fontsize=8)
            ax2.tick_params(axis="y", labelcolor=colors["branch_mispreds"])
            ax2.set_ylim(bottom=0)

        ax.plot(pcts_t, norm_times, color=colors["time"],
                marker="o", markersize=4, linewidth=1.8, label="Time (norm.)")
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("NNZ remaining (%)")
        ax.set_ylabel("Time (norm. to 100%)", color=colors["time"])
        ax.tick_params(axis="y", labelcolor=colors["time"])
        ax.set_title(matrix, fontsize=9, fontweight="bold")
        ax.invert_xaxis()
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Timing anomaly under random NNZ removal\n"
        "Blue = normalised SpMV time, Red dashed = branch misprediction rate",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = f"{OUT}/fig1_timing_anomaly_examples.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 – Branch mispred: random vs consec vs truncated
# ══════════════════════════════════════════════════════════════════════════

def fig_mispred_modes():
    papi = load_papi_modes("papi-perc-br-mispreds")

    # Matrices with the most dramatic random mispred increase
    dramatic = {m: max(d["random"][p]["branch_mispreds"] for p in d["random"])
                for m, d in papi.items() if "random" in d and "consec" in d}
    targets = sorted(dramatic, key=lambda m: -dramatic[m])[:6]

    mode_styles = {
        "random":    ("#d62728", "o-",  "Random removal"),
        "consec":    ("#2ca02c", "s--", "Consecutive removal"),
        "truncated": ("#ff7f0e", "^:", "Truncated removal"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for ax, matrix in zip(axes, targets):
        modes = papi[matrix]
        for mode, (color, style, label) in mode_styles.items():
            if mode not in modes:
                continue
            pcts, vals = sorted_series(modes[mode], "branch_mispreds")
            ax.plot(pcts, vals, style, color=color, linewidth=1.6,
                    markersize=4, label=label)
        ax.set_title(matrix, fontsize=9, fontweight="bold")
        ax.set_xlabel("NNZ remaining (%)")
        ax.set_ylabel("Branch mispred %")
        ax.invert_xaxis()
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Branch misprediction rate vs NNZ remaining\n"
        "Random removal drives mispredictions up; consecutive removal does not",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = f"{OUT}/fig2_mispred_random_vs_consec.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 3 – Distribution of slowdown factors (full 461-matrix sweep)
# ══════════════════════════════════════════════════════════════════════════

def fig_slowdown_distribution():
    timing = load_sweep("sweep_matrices_warmup", "timing")
    stats_list = slowdown_stats(timing)
    slowdowns = [s["slowdown"] for s in stats_list]

    n_anomalous   = sum(1 for x in slowdowns if x > 1.05)
    n_severe      = sum(1 for x in slowdowns if x > 2.0)
    n_very_severe = sum(1 for x in slowdowns if x > 3.0)
    n_total       = len(slowdowns)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.8, max(slowdowns) + 0.1, 40)
    ax.hist(slowdowns, bins=bins, color="#1f77b4", edgecolor="white", linewidth=0.4)
    ax.axvline(1.0, color="red",    linestyle="--", linewidth=1.5, label="No slowdown (1×)")
    ax.axvline(2.0, color="orange", linestyle="--", linewidth=1.5, label="2× slowdown")
    ax.axvline(3.0, color="darkred",linestyle="--", linewidth=1.5, label="3× slowdown")

    ax.set_xlabel("Peak slowdown factor (max time / time at 100% NNZ)")
    ax.set_ylabel("Number of matrices")
    ax.set_title(
        f"Distribution of timing slowdown under random NNZ removal\n"
        f"n={n_total} matrices  |  >1.05×: {n_anomalous} ({100*n_anomalous/n_total:.0f}%)  "
        f"|  >2×: {n_severe} ({100*n_severe/n_total:.0f}%)  "
        f"|  >3×: {n_very_severe} ({100*n_very_severe/n_total:.0f}%)")
    ax.legend()
    fig.tight_layout()
    path = f"{OUT}/fig3_slowdown_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 4 – Slowdown vs peak branch misprediction rate (scatter)
# ══════════════════════════════════════════════════════════════════════════

def fig_slowdown_vs_mispred():
    timing = load_sweep("sweep_matrices_warmup", "timing")
    papi   = load_sweep("sweep_matrices_warmup", "papi")

    common = sorted(set(timing) & set(papi))
    xs, ys = [], []
    for m in common:
        t100 = timing[m].get(100, {}).get("time")
        if t100 is None or t100 == 0:
            continue
        max_t = max(r["time"] for r in timing[m].values())
        slowdown = max_t / t100
        max_mp = max(r["branch_mispreds"] for r in papi[m].values())
        xs.append(max_mp)
        ys.append(slowdown)

    xs, ys = np.array(xs), np.array(ys)
    slope, intercept, r, p_val, _ = stats.linregress(xs, ys)
    x_line = np.linspace(xs.min(), xs.max(), 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, alpha=0.35, s=12, color="#1f77b4", label="Matrix")
    ax.plot(x_line, y_line, color="red", linewidth=1.8,
            label=f"Linear fit  r={r:.2f}, p={p_val:.2e}")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Peak branch misprediction rate (%) under random removal")
    ax.set_ylabel("Peak slowdown factor")
    ax.set_title(
        "Peak slowdown vs peak branch misprediction rate\n"
        f"n={len(xs)} SpMV matrices (sweep_matrices_warmup)")
    ax.legend()
    fig.tight_layout()
    path = f"{OUT}/fig4_slowdown_vs_mispred_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 5 – SpMV vs SpMM
# ══════════════════════════════════════════════════════════════════════════

def fig_spmv_vs_spmm():
    """
    spv8 has timing for 1185 SpMV matrices + 2 SpMM matrices (nemspmm1/2).
    Compare slowdown distributions.
    """
    spmm_names = {"nemspmm1", "nemspmm2"}
    timing_all = load_sweep("spv8", "timing")

    spmv_slowdowns, spmm_data = [], {}
    for matrix, data in timing_all.items():
        t100 = data.get(100, {}).get("time")
        if t100 is None or t100 == 0:
            continue
        max_t = max(r["time"] for r in data.values())
        sd = max_t / t100
        if matrix in spmm_names:
            pcts = sorted(data.keys(), reverse=True)
            spmm_data[matrix] = (pcts, [data[p]["time"] for p in pcts], sd)
        else:
            spmv_slowdowns.append(sd)

    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1])

    # Left: SpMV distribution
    ax1 = fig.add_subplot(gs[0])
    bins = np.linspace(0.8, max(spmv_slowdowns) + 0.05, 40)
    ax1.hist(spmv_slowdowns, bins=bins, color="#1f77b4", edgecolor="white", linewidth=0.4)
    ax1.axvline(1.0, color="red", linestyle="--", linewidth=1.4, label="1× (no anomaly)")
    n_anom = sum(1 for x in spmv_slowdowns if x > 1.05)
    ax1.set_xlabel("Peak slowdown factor")
    ax1.set_ylabel("Number of matrices")
    ax1.set_title(
        f"SpMV (n={len(spmv_slowdowns)} matrices)\n"
        f"{n_anom} ({100*n_anom/len(spmv_slowdowns):.0f}%) show slowdown > 1.05×")
    ax1.legend()

    # Right: SpMM timing curves
    ax2 = fig.add_subplot(gs[1])
    colors_spmm = {"nemspmm1": "#2ca02c", "nemspmm2": "#ff7f0e"}
    for matrix, (pcts, times, sd) in spmm_data.items():
        t100 = times[0]
        norm = [t / t100 for t in times]
        ax2.plot(pcts, norm, marker="o", markersize=5, linewidth=1.8,
                 color=colors_spmm[matrix], label=f"{matrix} (peak {sd:.2f}×)")
    ax2.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("NNZ remaining (%)")
    ax2.set_ylabel("Time (norm. to 100% NNZ)")
    ax2.invert_xaxis()
    ax2.set_title("SpMM matrices (nemspmm1, nemspmm2)\nMonotonically decreasing – no anomaly")
    ax2.legend()
    ax2.set_ylim(bottom=0)

    fig.suptitle("SpMV suffers timing anomaly; SpMM does not (limited data)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = f"{OUT}/fig5_spmv_vs_spmm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 6 – Custom SpMV vs MKL: both exhibit the anomaly
# ══════════════════════════════════════════════════════════════════════════

def fig_custom_vs_mkl():
    """Show that the anomaly persists even in MKL's optimised SpMV."""
    timing_custom = load_sweep("sweep_matrices_warmup", "timing")
    timing_mkl    = load_sweep("sweep_matrices_mkl",    "timing")

    common = sorted(set(timing_custom) & set(timing_mkl))

    def get_slowdown(data):
        t100 = data.get(100, {}).get("time")
        if not t100:
            return None
        return max(r["time"] for r in data.values()) / t100

    custom_sd, mkl_sd = [], []
    for m in common:
        cs = get_slowdown(timing_custom[m])
        ms = get_slowdown(timing_mkl[m])
        if cs is not None and ms is not None:
            custom_sd.append(cs)
            mkl_sd.append(ms)

    custom_sd, mkl_sd = np.array(custom_sd), np.array(mkl_sd)
    n_custom_anom = (custom_sd > 1.05).sum()
    n_mkl_anom    = (mkl_sd    > 1.05).sum()
    n = len(custom_sd)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bins = np.linspace(0.8, max(custom_sd.max(), mkl_sd.max()) + 0.1, 40)
    axes[0].hist(custom_sd, bins=bins, color="#1f77b4", edgecolor="white", linewidth=0.4)
    axes[0].axvline(1.0, color="red", linestyle="--", linewidth=1.4)
    axes[0].set_title(
        f"Custom CSR SpMV (n={n})\n{n_custom_anom} ({100*n_custom_anom/n:.0f}%) anomalous")
    axes[0].set_xlabel("Peak slowdown factor"); axes[0].set_ylabel("Matrices")

    axes[1].hist(mkl_sd, bins=bins, color="#ff7f0e", edgecolor="white", linewidth=0.4)
    axes[1].axvline(1.0, color="red", linestyle="--", linewidth=1.4)
    axes[1].set_title(
        f"Intel MKL SpMV (n={n})\n{n_mkl_anom} ({100*n_mkl_anom/n:.0f}%) anomalous")
    axes[1].set_xlabel("Peak slowdown factor"); axes[1].set_ylabel("Matrices")

    fig.suptitle(
        "Timing anomaly under random NNZ removal: Custom CSR vs Intel MKL\n"
        "Both implementations are affected — the anomaly is algorithmic, not implementation-specific",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = f"{OUT}/fig6_custom_vs_mkl.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 7 – Raw branch mispred counts: random vs consec (34 matrices)
# ══════════════════════════════════════════════════════════════════════════

def fig_raw_mispred_heatmap():
    """
    For all 34 deep-dive matrices show the difference in raw mispred counts
    (random − consec) as a heatmap across percentages.
    """
    papi = load_papi_modes("papi-raw-br-mispreds")

    matrices = sorted(papi.keys(), key=lambda m: m.lower())
    pcts = list(reversed(PERCENTAGES))  # x-axis: 10→100

    diff_matrix = []
    for m in matrices:
        row = []
        for p in pcts:
            r = papi[m].get("random", {}).get(p, {}).get("branch_mispreds", 0)
            c = papi[m].get("consec",  {}).get(p, {}).get("branch_mispreds", 0)
            row.append(r - c)
        diff_matrix.append(row)

    diff_matrix = np.array(diff_matrix)
    vmax = np.percentile(np.abs(diff_matrix), 95)

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(diff_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Extra mispreds (random − consec, raw count)")
    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels([f"{p}%" for p in pcts], fontsize=8)
    ax.set_yticks(range(len(matrices)))
    ax.set_yticklabels(matrices, fontsize=7)
    ax.set_xlabel("NNZ remaining (%)")
    ax.set_title(
        "Extra branch mispredictions introduced by random (vs consecutive) NNZ removal\n"
        "Red = random causes more mispredictions, Blue = fewer")
    fig.tight_layout()
    path = f"{OUT}/fig7_raw_mispred_diff_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 8 – Timing vs mispred side-by-side for the 34-matrix deep-dive
# ══════════════════════════════════════════════════════════════════════════

def fig_timing_mispred_panel():
    """
    For the 34 deep-dive matrices: timing (from sweep_matrices_warmup) +
    mispred % (from papi-perc-br-mispreds) for random vs consec.
    Show 6 most dramatic.
    """
    timing = load_sweep("sweep_matrices_warmup", "timing")
    papi   = load_papi_modes("papi-perc-br-mispreds")

    # Sort by max random mispred
    dramatic = {m: max(papi[m]["random"][p]["branch_mispreds"]
                       for p in papi[m]["random"])
                for m in papi if "random" in papi[m] and m in timing}
    targets = sorted(dramatic, key=lambda m: -dramatic[m])[:6]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, matrix in zip(axes, targets):
        ax2 = ax.twinx()

        # Timing (random only, since consec not available in sweep)
        if matrix in timing:
            pcts_t, times = sorted_series(timing[matrix], "time")
            t100 = times[0]
            norm_t = [t / t100 for t in times]
            ax.plot(pcts_t, norm_t, "o-", color="#1f77b4", linewidth=1.8,
                    markersize=4, label="Time (random, norm.)", zorder=4)
            ax.axhline(1.0, color="#1f77b4", linestyle=":", linewidth=0.7)

        # Branch mispred %: random vs consec
        colors2 = {"random": "#d62728", "consec": "#2ca02c", "truncated": "#ff7f0e"}
        for mode in ("random", "consec", "truncated"):
            if mode in papi.get(matrix, {}):
                pcts_p, vals = sorted_series(papi[matrix][mode], "branch_mispreds")
                ax2.plot(pcts_p, vals, "--", color=colors2[mode],
                         linewidth=1.4, markersize=3, marker="s",
                         label=f"Mispred {mode}")

        ax.set_xlabel("NNZ remaining (%)")
        ax.set_ylabel("Time (norm.)", color="#1f77b4")
        ax2.set_ylabel("Branch mispred %", color="#d62728")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax.invert_xaxis()
        ax.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        ax.set_title(matrix, fontsize=9, fontweight="bold")

        # Combined legend
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=6, loc="upper right")

    fig.suptitle(
        "Timing (blue) and branch misprediction % (dashed) for 6 matrices\n"
        "Random removal elevates mispreds and causes timing to peak mid-sweep;\n"
        "consecutive removal keeps mispreds near zero",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = f"{OUT}/fig8_timing_and_mispred_panel.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
#  SUMMARY STATS (printed, used in markdown)
# ══════════════════════════════════════════════════════════════════════════

def print_summary_stats():
    print("\n=== Summary statistics ===")

    # sweep_matrices_warmup
    timing_w = load_sweep("sweep_matrices_warmup", "timing")
    s_w = slowdown_stats(timing_w)
    n = len(s_w)
    print(f"\nsweep_matrices_warmup ({n} SpMV matrices):")
    print(f"  slowdown > 1.05× : {sum(1 for s in s_w if s['slowdown']>1.05)} ({100*sum(1 for s in s_w if s['slowdown']>1.05)/n:.0f}%)")
    print(f"  slowdown > 2×    : {sum(1 for s in s_w if s['slowdown']>2.0)} ({100*sum(1 for s in s_w if s['slowdown']>2.0)/n:.0f}%)")
    print(f"  slowdown > 3×    : {sum(1 for s in s_w if s['slowdown']>3.0)} ({100*sum(1 for s in s_w if s['slowdown']>3.0)/n:.0f}%)")
    worst = sorted(s_w, key=lambda s: -s["slowdown"])[:3]
    print(f"  worst: {[(s['matrix'], round(s['slowdown'],2)) for s in worst]}")

    # MKL
    timing_m = load_sweep("sweep_matrices_mkl", "timing")
    s_m = slowdown_stats(timing_m)
    nm = len(s_m)
    print(f"\nsweep_matrices_mkl ({nm} SpMV matrices, Intel MKL):")
    print(f"  slowdown > 1.05× : {sum(1 for s in s_m if s['slowdown']>1.05)} ({100*sum(1 for s in s_m if s['slowdown']>1.05)/nm:.0f}%)")

    # papi-perc-br-mispreds: random vs consec
    papi = load_papi_modes("papi-perc-br-mispreds")
    max_random = [max(papi[m]["random"][p]["branch_mispreds"]
                      for p in papi[m]["random"])
                  for m in papi if "random" in papi[m]]
    max_consec = [max(papi[m]["consec"][p]["branch_mispreds"]
                      for p in papi[m]["consec"])
                  for m in papi if "consec" in papi[m]]
    print(f"\npapi-perc-br-mispreds (34 deep-dive matrices):")
    print(f"  median peak mispred (random) : {np.median(max_random):.2f}%")
    print(f"  median peak mispred (consec) : {np.median(max_consec):.2f}%")
    print(f"  max random mispred: {max(max_random):.2f}%")
    print(f"  max consec mispred: {max(max_consec):.2f}%")

    # SpMM
    timing_spv8 = load_sweep("spv8", "timing")
    spmm = {m: timing_spv8[m] for m in ["nemspmm1", "nemspmm2"] if m in timing_spv8}
    for m, data in spmm.items():
        t100 = data[100]["time"]
        max_t = max(r["time"] for r in data.values())
        print(f"\nSpMM {m}: slowdown = {max_t/t100:.2f}×")

# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures...")
    fig_timing_examples()
    fig_mispred_modes()
    fig_slowdown_distribution()
    fig_slowdown_vs_mispred()
    fig_spmv_vs_spmm()
    fig_custom_vs_mkl()
    fig_raw_mispred_heatmap()
    fig_timing_mispred_panel()
    print_summary_stats()
    print(f"\nAll figures saved to {OUT}/")
