"""
empirical measurements.

Part 1 compares Heapsort against Quicksort and Merge Sort across input
distributions and sizes.

Part 2 measures how priority queue operations scale with the number of
tasks already in the queue, to check the O(log n) claim against a clock.

Writes sorting_results.csv and pq_results.csv.

"""

import csv
import random
import time

from heapsort import heapsort, merge_sort, quicksort
from priority_queue import MaxHeapPriorityQueue, Task


# ---------------------------------------------------------------------------
# Part 1: sorting
# ---------------------------------------------------------------------------
def make_array(n, kind, seed=42):
    rng = random.Random(seed)

    if kind == "random":
        return [rng.randint(0, 1_000_000) for _ in range(n)]
    if kind == "sorted":
        return list(range(n))
    if kind == "reverse":
        return list(range(n, 0, -1))
    if kind == "repeated":
        return [rng.randint(0, 9) for _ in range(n)]
    raise ValueError(f"unknown distribution: {kind}")


def time_sort(sort_fn, data, trials=7):
    """
    Median of several trials; each trial gets its own copy of the input.

    Seven rather than three: at n = 16,000 a three-trial median still
    swung by nearly 50 percent between runs, enough to invent a trend
    that was not there.
    """
    times = []
    for _ in range(trials):
        work = list(data)
        start = time.perf_counter()
        result = sort_fn(work)
        times.append(time.perf_counter() - start)

    ok = all(result[i] <= result[i + 1] for i in range(len(result) - 1))
    times.sort()
    return times[len(times) // 2], ok


def run_sorting_benchmark():
    sizes = [1000, 2000, 4000, 8000, 16000]
    kinds = ["random", "sorted", "reverse", "repeated"]
    algorithms = {
        "Heapsort": heapsort,
        "Quicksort": quicksort,
        "Merge Sort": merge_sort,
    }

    rows = []
    print("=" * 74)
    print("PART 1: SORTING")
    print("=" * 74)

    for kind in kinds:
        for n in sizes:
            data = make_array(n, kind)
            for name, fn in algorithms.items():
                secs, ok = time_sort(fn, data)
                rows.append([kind, n, name, f"{secs:.6f}",
                             "yes" if ok else "NO"])
                print(f"{kind:9} n={n:<7} {name:12} {secs:9.5f}s  sorted={ok}")
        print("-" * 74)

    with open("sorting_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distribution", "n", "algorithm", "time_seconds",
                    "correct"])
        w.writerows(rows)

    return rows


# ---------------------------------------------------------------------------
# Part 1b: Heapsort swap counts
# ---------------------------------------------------------------------------
def count_heapsort_swaps(arr):
    """
    Run Heapsort while counting swaps, split between the build phase and
    the extraction phase.

    Counting beats timing for the question of whether input ordering
    changes how much work Heapsort does, because a count is exact and
    repeatable while a stopwatch picks up cache state and OS scheduling.
    """
    a = list(arr)
    n = len(a)
    build_swaps = 0
    extract_swaps = 0

    def sift(i, size):
        swaps = 0
        while True:
            largest = i
            left, right = 2 * i + 1, 2 * i + 2
            if left < size and a[left] > a[largest]:
                largest = left
            if right < size and a[right] > a[largest]:
                largest = right
            if largest == i:
                return swaps
            a[i], a[largest] = a[largest], a[i]
            swaps += 1
            i = largest

    for i in range(n // 2 - 1, -1, -1):
        build_swaps += sift(i, n)

    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        extract_swaps += 1
        extract_swaps += sift(0, end)

    return build_swaps, extract_swaps


def run_swap_count_benchmark(n=16000):
    kinds = ["random", "sorted", "reverse", "repeated"]
    rows = []

    print("\n" + "=" * 74)
    print(f"PART 1b: HEAPSORT SWAP COUNTS (n = {n:,}, exact, not timed)")
    print("=" * 74)
    print(f"{'Distribution':<20}{'Build':>12}{'Extract':>12}{'Total':>12}")
    print("-" * 74)

    for kind in kinds:
        data = make_array(n, kind)
        build, extract = count_heapsort_swaps(data)
        rows.append([kind, n, build, extract, build + extract])
        print(f"{kind:<20}{build:>12,}{extract:>12,}{build + extract:>12,}")

    print("-" * 74)

    with open("swap_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distribution", "n", "build_swaps", "extract_swaps",
                    "total_swaps"])
        w.writerows(rows)

    return rows


# ---------------------------------------------------------------------------
# Part 2: priority queue operation scaling
# ---------------------------------------------------------------------------
def run_pq_benchmark():
    """
    Time each operation against a queue already holding n tasks.

    If the operations really are O(log n), then each time n is multiplied
    by four the cost should rise by a roughly constant additive amount
    rather than scaling with n. That is a far weaker signal than the
    quadrupling that marks a quadratic algorithm, so the numbers are
    reported as measured and read with that in mind.
    """
    sizes = [1000, 4000, 16000, 64000, 256000]
    ops = 2000
    rows = []

    print("\n" + "=" * 74)
    print("PART 2: PRIORITY QUEUE OPERATIONS (microseconds per operation)")
    print("=" * 74)
    print(f"{'n':>8}  {'insert':>10}  {'extract_max':>12}  "
          f"{'increase_key':>13}  {'is_empty':>10}")
    print("-" * 74)

    for n in sizes:
        rng = random.Random(11)

        pq = MaxHeapPriorityQueue()
        for i in range(n):
            pq.insert(Task(f"P{i}", priority=rng.randint(0, 10**6),
                           arrival_time=i))

        # insert
        start = time.perf_counter()
        for i in range(ops):
            pq.insert(Task(f"N{i}", priority=rng.randint(0, 10**6),
                           arrival_time=n + i))
        insert_us = (time.perf_counter() - start) / ops * 1e6

        # increase_key on tasks that are present. The new priority has to
        # climb monotonically, since the same task can be picked twice and
        # increase_key rejects a value below the current one.
        targets = [f"P{rng.randrange(n)}" for _ in range(ops)]
        start = time.perf_counter()
        for step, tid in enumerate(targets):
            if tid in pq:
                pq.increase_key(tid, 10**6 + step)
        increase_us = (time.perf_counter() - start) / ops * 1e6

        # extract_max
        start = time.perf_counter()
        for _ in range(ops):
            pq.extract_max()
        extract_us = (time.perf_counter() - start) / ops * 1e6

        # is_empty, expected to be flat
        start = time.perf_counter()
        for _ in range(ops):
            pq.is_empty()
        empty_us = (time.perf_counter() - start) / ops * 1e6

        rows.append([n, f"{insert_us:.4f}", f"{extract_us:.4f}",
                     f"{increase_us:.4f}", f"{empty_us:.4f}"])
        print(f"{n:>8}  {insert_us:>10.4f}  {extract_us:>12.4f}  "
              f"{increase_us:>13.4f}  {empty_us:>10.4f}")

    with open("pq_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "insert_us", "extract_max_us", "increase_key_us",
                    "is_empty_us"])
        w.writerows(rows)

    return rows


if __name__ == "__main__":
    run_sorting_benchmark()
    run_swap_count_benchmark()
    run_pq_benchmark()
    print("\nWrote sorting_results.csv, swap_counts.csv, and pq_results.csv")
