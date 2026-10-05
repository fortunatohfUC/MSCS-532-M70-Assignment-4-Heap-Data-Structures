# MSCS-532 Assignment 4: Heap Data Structures

Heapsort and a max-heap priority queue in Python, with a scheduler simulation
and empirical analysis.

## Running the code

Python 3.8+. Only `make_charts.py` needs a third-party package.

```bash
python3 heapsort.py         # 11 sorting correctness checks
python3 priority_queue.py   # 18 priority queue checks, incl. a stress test
python3 scheduler.py        # scheduler simulation
python3 benchmarks.py       # reproduces all measurements (~3 min)

pip3 install matplotlib
python3 make_charts.py      # rebuilds the figures
```

Everything is seeded, so swap counts, scheduler metrics, and correctness are
reproducible; timings vary by machine.

## Using the code

```python
from heapsort import heapsort
heapsort([5, 2, 9, 2, 7])          # sorts in place -> [2, 2, 5, 7, 9]

from priority_queue import MaxHeapPriorityQueue, Task
pq = MaxHeapPriorityQueue()
pq.insert(Task("T1", priority=3, arrival_time=0, deadline=20, duration=4))
pq.increase_key("T1", 9)           # O(log n)
pq.extract_max()                   # highest priority task
pq.is_empty()                      # O(1)
```

## Findings

### Heapsort

Times at n = 16,000, median of seven trials:

| Distribution | Heapsort | Quicksort | Merge Sort |
|---|---|---|---|
| Random | 0.02523 s | 0.01856 s | 0.02143 s |
| Sorted | 0.02697 s | 0.01625 s | 0.01402 s |
| Reverse-sorted | 0.02238 s | 0.01801 s | 0.01476 s |
| Repeated elements | 0.02103 s | **0.65511 s** | 0.02624 s |

- **Θ(n log n) in every case, and the counts prove it.** Swap counts at
  n = 16,000 vary only from 189,552 to 220,380 — a 16% spread — across
  inputs that swing Quicksort by more than fortyfold. The extraction phase
  is ~95% of all swaps, which is the O(n) build / O(n log n) extraction
  split showing up as measured numbers.
- **It loses on random data anyway.** 36% slower than Quicksort despite the
  same complexity class, from cache behavior (sift-down jumps i → 2i → 4i)
  and roughly double the comparisons per level.
- **It wins outright on duplicates.** 31× faster than Quicksort on repeated
  elements, where Lomuto partitioning goes quadratic and no pivot choice
  helps. Duplicates make sift-down stop *earlier*.
- **One honest exception to "O(n log n) always":** on all-identical input
  Heapsort does exactly n−1 swaps and zero sift swaps — Θ(n), not
  Θ(n log n), because sift-down stops immediately when no child is strictly
  greater.
- **O(1) auxiliary space**, which required writing sift-down iteratively;
  the recursive version would cost O(log n) of stack.

### Priority Queue

Array-backed max-heap over `Task` objects. A dict from task ID to heap index
keeps `increase_key`/`decrease_key` at O(log n) — without it, finding the task
would be an O(n) scan.

| Operation | Complexity |
|---|---|
| `insert` | O(log n) worst, ~O(1) expected for random priorities |
| `extract_max` | O(log n) |
| `increase_key` / `decrease_key` | O(log n), O(1) lookup + one-directional sift |
| `is_empty` / `peek` | O(1) |

Measured across a 256× increase in queue size (1,000 → 256,000 tasks):
`is_empty` stayed flat at 0.055–0.070 µs, confirming O(1). `extract_max` grew
only 4.7×, against 256× for linear and ~1.8× for pure log n — logarithmic in
shape, with the per-level constant drifting up as the heap outgrows cache.

### Scheduler simulation

2,000 tasks, 5 priority levels, 85% processor utilization, non-preemptive.
Priority queue vs FCFS on an identical task stream:

| Metric | Priority queue | FCFS |
|---|---|---|
| Average waiting time | 16.14 | 16.05 |
| Maximum waiting time | **286** | 82 |
| Deadline miss rate | **11.7%** | 17.4% |

**Priority scheduling does not reduce total waiting time — it redistributes
it.** The averages are identical to within half a percent, because both
schedulers are work-conserving and get through the same work in the same time.
What changes is who waits: highest-priority tasks go from 16.39 to 3.35
(4.9× faster), lowest-priority from 14.81 to 44.10 (3.0× slower).

That trade pays off here — deadline misses drop 33%, since the tasks with the
tightest deadlines are the ones served first. The cost is the tail: maximum
wait is 3.5× worse, which is starvation. Aging would fix it, and `increase_key`
already supports it; I left it off to measure the unmitigated behavior.
