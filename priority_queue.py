"""
MSCS-532 Assignment 4, Part 2: Priority Queue

A max-heap priority queue over Task objects, backed by a Python list.

Design decisions, with the reasoning kept here so the code and the report
agree:

  Array over pointers. A binary heap is a complete binary tree, so every
  level except the last is full and the last fills left to right. That
  shape means the tree maps onto contiguous indices with no gaps, and a
  node's children are found by arithmetic (2i+1, 2i+2) instead of by
  following pointers. No per-node child or parent references to store,
  and the sequential layout is far friendlier to the CPU cache than
  scattered node objects. A pointer-based tree would need explicit
  bookkeeping just to find the insertion point and the last node.

  Max-heap over min-heap. The scheduler here runs the highest-priority
  ready task first, so extract_max is the operation that must be cheap.
  A min-heap would be the right call for earliest-deadline-first, where
  the smallest deadline wins.

  Position map. increase_key and decrease_key take a task, and without
  help the heap would have to scan all n slots to find where that task
  currently sits, making the operation O(n) even though the sift itself
  is O(log n). A dict from task_id to heap index keeps the lookup O(1)
  and the whole operation O(log n). The map is updated on every swap.

"""


class Task:
    """
    One unit of schedulable work.

    Priority is what the heap orders on. Arrival time and deadline are
    carried for the scheduler simulation and do not affect heap ordering.
    """

    __slots__ = ("task_id", "priority", "arrival_time", "deadline",
                 "duration", "start_time", "finish_time")

    def __init__(self, task_id, priority, arrival_time=0, deadline=None,
                 duration=1):
        self.task_id = task_id
        self.priority = priority
        self.arrival_time = arrival_time
        self.deadline = deadline
        self.duration = duration

        # Filled in by the scheduler.
        self.start_time = None
        self.finish_time = None

    def waiting_time(self):
        """Time between arriving and actually starting."""
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time

    def missed_deadline(self):
        if self.deadline is None or self.finish_time is None:
            return False
        return self.finish_time > self.deadline

    def __repr__(self):
        return (f"Task(id={self.task_id}, priority={self.priority}, "
                f"arrival={self.arrival_time}, deadline={self.deadline})")


class MaxHeapPriorityQueue:
    """
    Max-heap priority queue. The task with the highest priority comes out
    first; ties are broken by earlier arrival time, so equal-priority work
    is served first come first served rather than arbitrarily.
    """

    def __init__(self):
        self._heap = []          # the array that holds the tree
        self._index = {}         # task_id -> position in self._heap

    # -----------------------------------------------------------------
    # Index arithmetic
    # -----------------------------------------------------------------
    @staticmethod
    def _parent(i):
        return (i - 1) // 2

    @staticmethod
    def _left(i):
        return 2 * i + 1

    @staticmethod
    def _right(i):
        return 2 * i + 2

    def _higher(self, i, j):
        """
        True when the task at i outranks the one at j.

        Higher priority wins. On a tie, the earlier arrival wins, which
        keeps the ordering deterministic and gives FCFS behavior within a
        priority level.
        """
        a, b = self._heap[i], self._heap[j]
        if a.priority != b.priority:
            return a.priority > b.priority
        return a.arrival_time < b.arrival_time

    def _swap(self, i, j):
        """Swap two slots and keep the position map in step."""
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._index[self._heap[i].task_id] = i
        self._index[self._heap[j].task_id] = j

    # -----------------------------------------------------------------
    # Sift operations
    # -----------------------------------------------------------------
    def _sift_up(self, i):
        """
        Move a too-high-priority element toward the root.

        O(log n): each step climbs one level, and the tree height is
        floor(log2 n).
        """
        while i > 0 and self._higher(i, self._parent(i)):
            parent = self._parent(i)
            self._swap(i, parent)
            i = parent

    def _sift_down(self, i):
        """
        Move a too-low-priority element toward the leaves.

        O(log n) for the same reason, in the other direction. Written
        iteratively so no stack frames are consumed.
        """
        n = len(self._heap)
        while True:
            best = i
            left, right = self._left(i), self._right(i)

            if left < n and self._higher(left, best):
                best = left
            if right < n and self._higher(right, best):
                best = right

            if best == i:
                return

            self._swap(i, best)
            i = best

    # -----------------------------------------------------------------
    # Core operations
    # -----------------------------------------------------------------
    def insert(self, task):
        """
        Add a task.

        Append to the end of the array, which keeps the tree complete,
        then sift up until the heap property holds again.

        O(log n). The append is amortized O(1) on a Python list, and the
        sift travels at most the height of the tree.
        """
        if task.task_id in self._index:
            raise KeyError(f"task {task.task_id} is already in the queue")

        self._heap.append(task)
        i = len(self._heap) - 1
        self._index[task.task_id] = i
        self._sift_up(i)

    def extract_max(self):
        """
        Remove and return the highest-priority task.

        The root is the answer. Move the last element into the root to
        keep the tree complete, then sift it down.

        O(log n), dominated by the sift. Returns None on an empty queue.
        """
        if not self._heap:
            return None

        top = self._heap[0]
        del self._index[top.task_id]

        last = self._heap.pop()
        if self._heap:                   # queue was not emptied by the pop
            self._heap[0] = last
            self._index[last.task_id] = 0
            self._sift_down(0)

        return top

    def peek(self):
        """Look at the highest-priority task without removing it. O(1)."""
        return self._heap[0] if self._heap else None

    def increase_key(self, task_id, new_priority):
        """
        Raise a task's priority and move it up to where it belongs.

        O(log n): the position map finds the task in O(1), and only a
        sift up is needed, because raising a priority can never push an
        element downward.
        """
        if task_id not in self._index:
            raise KeyError(f"task {task_id} is not in the queue")

        i = self._index[task_id]
        task = self._heap[i]

        if new_priority < task.priority:
            raise ValueError("new priority is lower; use decrease_key")

        task.priority = new_priority
        self._sift_up(i)

    def decrease_key(self, task_id, new_priority):
        """
        Lower a task's priority and move it down.

        O(log n), the mirror image of increase_key: lowering a priority
        can only push an element toward the leaves.
        """
        if task_id not in self._index:
            raise KeyError(f"task {task_id} is not in the queue")

        i = self._index[task_id]
        task = self._heap[i]

        if new_priority > task.priority:
            raise ValueError("new priority is higher; use increase_key")

        task.priority = new_priority
        self._sift_down(i)

    def is_empty(self):
        """O(1)."""
        return len(self._heap) == 0

    # -----------------------------------------------------------------
    # Inspection
    # -----------------------------------------------------------------
    def __len__(self):
        return len(self._heap)

    def __contains__(self, task_id):
        return task_id in self._index

    def is_valid_heap(self):
        """
        Check the heap property and the position map together.

        Used by the tests. A position map that drifts out of sync is the
        failure mode this design is most exposed to, so it gets checked
        explicitly rather than assumed.
        """
        n = len(self._heap)

        for i in range(n):
            if self._index.get(self._heap[i].task_id) != i:
                return False

        for i in range(n // 2):
            for child in (self._left(i), self._right(i)):
                if child < n and self._higher(child, i):
                    return False

        return len(self._index) == n

    def __repr__(self):
        return f"MaxHeapPriorityQueue(size={len(self._heap)})"


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------
def _run_checks():
    import random
    results = []

    # Empty queue
    pq = MaxHeapPriorityQueue()
    results.append(("empty queue is_empty", pq.is_empty()))
    results.append(("empty extract returns None", pq.extract_max() is None))
    results.append(("empty peek returns None", pq.peek() is None))

    # Single task
    pq.insert(Task("t1", priority=5))
    results.append(("insert makes non-empty", not pq.is_empty()))
    results.append(("peek finds it", pq.peek().task_id == "t1"))
    results.append(("extract returns it", pq.extract_max().task_id == "t1"))
    results.append(("empty again", pq.is_empty()))

    # Ordering: tasks must come out in descending priority
    pq = MaxHeapPriorityQueue()
    priorities = [3, 7, 1, 9, 4, 9, 2, 8]
    for i, p in enumerate(priorities):
        pq.insert(Task(f"t{i}", priority=p, arrival_time=i))
    out = []
    while not pq.is_empty():
        out.append(pq.extract_max().priority)
    results.append(("extracts in descending order",
                    out == sorted(priorities, reverse=True)))

    # Tie-break: equal priority, earlier arrival first
    pq = MaxHeapPriorityQueue()
    pq.insert(Task("late", priority=5, arrival_time=10))
    pq.insert(Task("early", priority=5, arrival_time=1))
    results.append(("ties break by arrival time",
                    pq.extract_max().task_id == "early"))

    # increase_key
    pq = MaxHeapPriorityQueue()
    for i in range(10):
        pq.insert(Task(f"t{i}", priority=i, arrival_time=i))
    pq.increase_key("t0", 100)
    results.append(("increase_key promotes to front",
                    pq.peek().task_id == "t0"))
    results.append(("heap still valid after increase", pq.is_valid_heap()))

    # decrease_key
    pq.decrease_key("t0", -100)
    results.append(("decrease_key demotes", pq.peek().task_id != "t0"))
    results.append(("heap still valid after decrease", pq.is_valid_heap()))

    # Guard rails
    try:
        pq.increase_key("t1", -999)
        results.append(("increase_key rejects a lower value", False))
    except ValueError:
        results.append(("increase_key rejects a lower value", True))

    try:
        pq.increase_key("nope", 5)
        results.append(("unknown task raises", False))
    except KeyError:
        results.append(("unknown task raises", True))

    try:
        pq.insert(Task("t1", priority=1))
        results.append(("duplicate id raises", False))
    except KeyError:
        results.append(("duplicate id raises", True))

    # Randomized stress: heap property and position map must survive
    rng = random.Random(7)
    pq = MaxHeapPriorityQueue()
    live = {}
    valid = True
    for step in range(3000):
        action = rng.random()
        if action < 0.45 or not live:
            tid = f"s{step}"
            p = rng.randint(0, 1000)
            pq.insert(Task(tid, priority=p, arrival_time=step))
            live[tid] = p
        elif action < 0.70:
            t = pq.extract_max()
            if t is not None:
                live.pop(t.task_id, None)
        else:
            tid = rng.choice(list(live))
            cur = live[tid]
            if rng.random() < 0.5:
                new = cur + rng.randint(0, 100)
                pq.increase_key(tid, new)
            else:
                new = cur - rng.randint(0, 100)
                pq.decrease_key(tid, new)
            live[tid] = new

        if step % 100 == 0 and not pq.is_valid_heap():
            valid = False
            break

    results.append(("3,000-step random stress stays valid",
                    valid and pq.is_valid_heap()))

    # Extraction order after all that churn must still be descending
    seq = []
    while not pq.is_empty():
        seq.append(pq.extract_max().priority)
    results.append(("post-stress order is descending",
                    all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))))

    print("Priority queue checks")
    print("-" * 56)
    for name, ok in results:
        print(f"  {name:38} {'PASS' if ok else 'FAIL'}")
    all_ok = all(ok for _, ok in results)
    print("-" * 56)
    print("all passed" if all_ok else "SOME CASES FAILED")
    return all_ok


if __name__ == "__main__":
    _run_checks()
