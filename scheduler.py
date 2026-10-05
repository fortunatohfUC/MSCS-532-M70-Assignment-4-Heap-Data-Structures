"""
MSCS-532 Assignment 4: Scheduler simulation

Runs a non-preemptive single-processor scheduler two ways over the same
workload, so the priority queue has something to be measured against:

  Priority scheduling  - highest priority ready task runs next
                         (uses MaxHeapPriorityQueue)
  First-come, first-served - earliest arrival runs next
                         (uses a plain FIFO queue)

Both see an identical task stream. The only difference is the order the
scheduler picks from, which is exactly what the priority queue buys.

"""

import csv
import random
from collections import deque

from priority_queue import MaxHeapPriorityQueue, Task


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------
def generate_workload(n_tasks, seed=42, utilization=0.85,
                      n_priority_levels=5):
    """
    Build a stream of tasks with staggered arrivals.

    The arrival window is derived from the total work so the processor runs
    at a chosen utilization. This matters: if arrivals are packed tighter
    than the processor can drain them, the queue grows without bound, every
    task misses its deadline, and the comparison between schedulers stops
    being informative because both are simply saturated. At 0.85 there is
    real queueing but the system keeps up.

    Deadlines scale with priority, so urgent work gets a tighter window.
    """
    rng = random.Random(seed)

    durations = [rng.randint(1, 10) for _ in range(n_tasks)]
    total_work = sum(durations)
    max_arrival = int(total_work / utilization)

    tasks = []
    for i in range(n_tasks):
        priority = rng.randint(1, n_priority_levels)
        arrival = rng.randint(0, max_arrival)
        duration = durations[i]

        # Higher priority gets a tighter deadline relative to its work.
        slack = duration * (n_priority_levels + 2 - priority) * 2
        deadline = arrival + duration + slack

        tasks.append(Task(task_id=f"T{i:05d}", priority=priority,
                          arrival_time=arrival, deadline=deadline,
                          duration=duration))

    tasks.sort(key=lambda t: t.arrival_time)
    return tasks


def _fresh_copies(tasks):
    """
    Each scheduler needs its own Task objects, since the simulation writes
    start_time and finish_time back onto them.
    """
    return [Task(t.task_id, t.priority, t.arrival_time, t.deadline,
                 t.duration) for t in tasks]


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------
def run_priority_scheduler(tasks):
    """
    Non-preemptive priority scheduling backed by the max-heap.

    The clock advances to the next arrival whenever nothing is ready. Each
    completed task records when it started and finished so waiting time and
    deadline misses can be computed afterward.

    Cost is O(n log n) overall: every task is inserted once and extracted
    once, and both are O(log n).
    """
    tasks = _fresh_copies(tasks)
    pq = MaxHeapPriorityQueue()

    clock = 0
    next_arrival = 0
    completed = []

    while next_arrival < len(tasks) or not pq.is_empty():
        # Admit everything that has arrived by now.
        while (next_arrival < len(tasks)
               and tasks[next_arrival].arrival_time <= clock):
            pq.insert(tasks[next_arrival])
            next_arrival += 1

        if pq.is_empty():
            # Processor idle: jump forward to the next arrival.
            clock = tasks[next_arrival].arrival_time
            continue

        task = pq.extract_max()
        task.start_time = clock
        clock += task.duration
        task.finish_time = clock
        completed.append(task)

    return completed


def run_fcfs_scheduler(tasks):
    """
    First-come, first-served, for comparison. Same loop, but the ready set
    is a FIFO queue, so arrival order decides everything and priority is
    ignored.
    """
    tasks = _fresh_copies(tasks)
    queue = deque()

    clock = 0
    next_arrival = 0
    completed = []

    while next_arrival < len(tasks) or queue:
        while (next_arrival < len(tasks)
               and tasks[next_arrival].arrival_time <= clock):
            queue.append(tasks[next_arrival])
            next_arrival += 1

        if not queue:
            clock = tasks[next_arrival].arrival_time
            continue

        task = queue.popleft()
        task.start_time = clock
        clock += task.duration
        task.finish_time = clock
        completed.append(task)

    return completed


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def summarize(completed, n_priority_levels=5):
    """Average waiting time overall, per priority level, and miss rate."""
    waits = [t.waiting_time() for t in completed]
    misses = sum(1 for t in completed if t.missed_deadline())

    by_priority = {}
    for level in range(1, n_priority_levels + 1):
        group = [t for t in completed if t.priority == level]
        if group:
            by_priority[level] = sum(t.waiting_time() for t in group) / len(group)

    return {
        "tasks": len(completed),
        "avg_wait": sum(waits) / len(waits) if waits else 0.0,
        "max_wait": max(waits) if waits else 0,
        "missed": misses,
        "miss_rate": misses / len(completed) if completed else 0.0,
        "wait_by_priority": by_priority,
    }


def run_simulation(n_tasks=2000, seed=42, verbose=True):
    """Run both schedulers over one workload and report the difference."""
    levels = 5
    tasks = generate_workload(n_tasks, seed=seed, n_priority_levels=levels)

    pri = summarize(run_priority_scheduler(tasks), levels)
    fcfs = summarize(run_fcfs_scheduler(tasks), levels)

    if verbose:
        print("=" * 72)
        print(f"SCHEDULER SIMULATION  ({n_tasks} tasks, "
              f"priority 1 = lowest, {levels} = highest)")
        print("=" * 72)
        print(f"{'Metric':<34}{'Priority queue':>18}{'FCFS':>18}")
        print("-" * 72)
        print(f"{'Average waiting time':<34}"
              f"{pri['avg_wait']:>18.2f}{fcfs['avg_wait']:>18.2f}")
        print(f"{'Maximum waiting time':<34}"
              f"{pri['max_wait']:>18}{fcfs['max_wait']:>18}")
        print(f"{'Deadline misses':<34}"
              f"{pri['missed']:>18}{fcfs['missed']:>18}")
        print(f"{'Deadline miss rate':<34}"
              f"{pri['miss_rate']:>17.1%}{fcfs['miss_rate']:>18.1%}")
        print("-" * 72)
        print("Average waiting time by priority level")
        print(f"{'Priority':<34}{'Priority queue':>18}{'FCFS':>18}")
        for level in sorted(pri["wait_by_priority"], reverse=True):
            label = f"  {level}" + ("  (highest)" if level == levels else
                                    "  (lowest)" if level == 1 else "")
            print(f"{label:<34}"
                  f"{pri['wait_by_priority'][level]:>18.2f}"
                  f"{fcfs['wait_by_priority'][level]:>18.2f}")
        print("-" * 72)

    return pri, fcfs


def write_results(pri, fcfs, levels=5):
    with open("scheduler_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "priority_queue", "fcfs"])
        w.writerow(["avg_wait", f"{pri['avg_wait']:.4f}",
                    f"{fcfs['avg_wait']:.4f}"])
        w.writerow(["max_wait", pri["max_wait"], fcfs["max_wait"]])
        w.writerow(["missed_deadlines", pri["missed"], fcfs["missed"]])
        w.writerow(["miss_rate", f"{pri['miss_rate']:.4f}",
                    f"{fcfs['miss_rate']:.4f}"])
        for level in sorted(pri["wait_by_priority"]):
            w.writerow([f"avg_wait_priority_{level}",
                        f"{pri['wait_by_priority'][level]:.4f}",
                        f"{fcfs['wait_by_priority'][level]:.4f}"])
    print("\nWrote scheduler_results.csv")


if __name__ == "__main__":
    p, f = run_simulation()
    write_results(p, f)
