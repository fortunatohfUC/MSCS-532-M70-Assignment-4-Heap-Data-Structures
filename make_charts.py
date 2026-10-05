"""
Generate the figures for MSCS-532 Assignment 4.

Reads sorting_results.csv, pq_results.csv, and scheduler_results.csv.


"""

import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "Heapsort": "#2B6CB0",
    "Quicksort": "#C05621",
    "Merge Sort": "#2F855A",
}
MARKERS = {"Heapsort": "o", "Quicksort": "s", "Merge Sort": "^"}
TITLES = {
    "random": "Random",
    "sorted": "Sorted",
    "reverse": "Reverse-sorted",
    "repeated": "Repeated elements",
}


def chart_sorting():
    data = defaultdict(lambda: defaultdict(list))
    with open("sorting_results.csv") as f:
        for row in csv.DictReader(f):
            data[row["distribution"]][row["algorithm"]].append(
                (int(row["n"]), float(row["time_seconds"])))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)

    for ax, kind in zip(axes, ["random", "sorted", "reverse", "repeated"]):
        for algo, points in data[kind].items():
            points.sort()
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    marker=MARKERS[algo], color=COLORS[algo], label=algo,
                    linewidth=2, markersize=6)
        ax.set_title(TITLES[kind], fontsize=12)
        ax.set_xlabel("Input size (n)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xticks([1000, 4000, 16000])
        ax.set_xticklabels(["1k", "4k", "16k"])

    axes[0].set_ylabel("Time (seconds, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Heapsort versus Quicksort and Merge Sort across four "
                 "input distributions", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig("figure1_sorting.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figure1_sorting.png")


def chart_pq():
    rows = []
    with open("pq_results.csv") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ns = [r["n"] for r in rows]

    for key, label, color, marker in [
        ("extract_max_us", "extract_max", "#C05621", "s"),
        ("increase_key_us", "increase_key", "#2F855A", "^"),
        ("insert_us", "insert", "#2B6CB0", "o"),
        ("is_empty_us", "is_empty", "#718096", "D"),
    ]:
        ax.plot(ns, [r[key] for r in rows], marker=marker, color=color,
                label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Tasks already in the queue (n)")
    ax.set_ylabel("Time per operation (microseconds)")
    ax.set_title("Priority queue operation cost as the queue grows",
                 fontsize=12)
    ax.set_xscale("log", base=4)
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{int(n/1000)}k" for n in ns])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("figure2_pq_operations.png", dpi=200)
    plt.close(fig)
    print("wrote figure2_pq_operations.png")


def chart_scheduler():
    waits = {}
    with open("scheduler_results.csv") as f:
        for row in csv.DictReader(f):
            if row["metric"].startswith("avg_wait_priority_"):
                level = int(row["metric"].rsplit("_", 1)[1])
                waits[level] = (float(row["priority_queue"]),
                                float(row["fcfs"]))

    levels = sorted(waits)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))

    width = 0.38
    pos = range(len(levels))
    pq_vals = [waits[l][0] for l in levels]
    fc_vals = [waits[l][1] for l in levels]

    b1 = ax.bar([p - width / 2 for p in pos], pq_vals, width,
                label="Priority queue", color="#2B6CB0")
    b2 = ax.bar([p + width / 2 for p in pos], fc_vals, width,
                label="FCFS", color="#C05621")

    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                    f"{b.get_height():.1f}", ha="center", fontsize=9)

    ax.set_xticks(list(pos))
    ax.set_xticklabels([f"{l}" + (" (highest)" if l == max(levels)
                                  else " (lowest)" if l == min(levels)
                                  else "") for l in levels])
    ax.set_xlabel("Task priority level")
    ax.set_ylabel("Average waiting time")
    ax.set_title("Waiting time by priority: the queue redistributes delay, "
                 "it does not remove it", fontsize=11)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("figure3_scheduler.png", dpi=200)
    plt.close(fig)
    print("wrote figure3_scheduler.png")


if __name__ == "__main__":
    chart_sorting()
    chart_pq()
    chart_scheduler()
