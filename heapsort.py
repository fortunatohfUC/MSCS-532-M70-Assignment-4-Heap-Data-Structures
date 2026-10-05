"""
Part 1: Heapsort

Heapsort built on an array-backed max-heap, plus Merge Sort and randomized
Quicksort for the empirical comparison.

The heap lives in the same array being sorted. For a node at index i:
    parent(i) = (i - 1) // 2
    left(i)   = 2i + 1
    right(i)  = 2i + 2

"""

import random
import sys

sys.setrecursionlimit(20000)


# ---------------------------------------------------------------------------
# Heapsort
# ---------------------------------------------------------------------------
def _sift_down(arr, i, heap_size):
    """
    Restore the max-heap property at index i, assuming both subtrees of i
    are already valid max-heaps.

    Written as a loop rather than a recursion. The recursive form is the
    textbook presentation, but it adds O(log n) stack frames per call and
    would make the "constant extra space" claim false in practice.

    Cost is O(log n): each iteration moves one level down a tree of height
    floor(log2 n).
    """
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < heap_size and arr[left] > arr[largest]:
            largest = left
        if right < heap_size and arr[right] > arr[largest]:
            largest = right

        if largest == i:
            return                      # heap property already holds

        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest                     # follow the element down


def build_max_heap(arr):
    """
    Turn an arbitrary array into a max-heap in place.

    Starts from the last internal node and sifts down toward the root.
    Everything from index n//2 onward is a leaf, and a leaf is already a
    valid heap of size one, so those are skipped.

    This is O(n), not O(n log n). Most nodes sit near the bottom of the
    tree and have almost no distance to travel: there are n/2 nodes at
    height 0, n/4 at height 1, and so on. Summing height times node count
    over all levels gives n * sum(h / 2^h), and that series converges to 2.
    """
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down(arr, i, n)
    return arr


def heapsort(arr):
    """
    Sort ascending, in place, using a max-heap.

    Two phases. Build the max-heap in O(n), then repeatedly swap the root
    (the largest remaining element) to the end of the unsorted region and
    sift the new root down over a heap one element smaller. After n-1
    extractions the array is sorted.

    Theta(n log n) on every input of distinct elements, with O(1) extra
    space. Nothing about the algorithm depends on how the input was
    ordered, which is the property that distinguishes it from Quicksort.
    """
    n = len(arr)
    if n <= 1:
        return arr

    build_max_heap(arr)

    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]   # largest goes to its final spot
        _sift_down(arr, 0, end)               # restore the smaller heap

    return arr


# ---------------------------------------------------------------------------
# Merge Sort, for comparison
# ---------------------------------------------------------------------------
def merge_sort(arr):
    """
    Stable divide-and-conquer sort. Theta(n log n) on every input, but it
    needs O(n) auxiliary space for the merge.
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left, right):
    merged = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:          # <= keeps the sort stable
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


# ---------------------------------------------------------------------------
# Randomized Quicksort, for comparison
# ---------------------------------------------------------------------------
def quicksort(arr):
    """
    In-place Quicksort with a uniformly random pivot.

    Recurses into the smaller partition and loops on the larger one, which
    bounds the stack at O(log n) without changing the running time.
    """
    def sort(low, high):
        while low < high:
            p = _partition_random(arr, low, high)
            if p - low < high - p:
                sort(low, p - 1)
                low = p + 1
            else:
                sort(p + 1, high)
                high = p - 1

    if arr:
        sort(0, len(arr) - 1)
    return arr


def _partition_random(arr, low, high):
    k = random.randint(low, high)
    arr[k], arr[high] = arr[high], arr[k]
    return _partition(arr, low, high)


def _partition(arr, low, high):
    """Lomuto partition around arr[high]."""
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------
def _is_max_heap(arr, heap_size=None):
    """Verify the max-heap property holds for every internal node."""
    n = heap_size if heap_size is not None else len(arr)
    for i in range(n // 2):
        left, right = 2 * i + 1, 2 * i + 2
        if left < n and arr[i] < arr[left]:
            return False
        if right < n and arr[i] < arr[right]:
            return False
    return True


def _run_checks():
    cases = {
        "empty array": [],
        "single element": [42],
        "two sorted": [1, 2],
        "two reversed": [2, 1],
        "all identical": [7] * 50,
        "already sorted": list(range(50)),
        "reverse sorted": list(range(50, 0, -1)),
        "many duplicates": [random.randint(0, 4) for _ in range(50)],
        "negatives": [random.randint(-50, 50) for _ in range(50)],
        "random": [random.randint(0, 1000) for _ in range(200)],
    }

    print("Sorting checks")
    print("-" * 56)
    ok_all = True

    for name, data in cases.items():
        expected = sorted(data)
        results = {
            "heapsort": heapsort(list(data)),
            "merge_sort": merge_sort(list(data)),
            "quicksort": quicksort(list(data)),
        }
        ok = all(r == expected for r in results.values())
        ok_all = ok_all and ok
        print(f"  {name:22} {'PASS' if ok else 'FAIL'}")

    # build_max_heap must actually produce a heap, not just happen to work
    print("-" * 56)
    heap_ok = True
    for trial in range(200):
        data = [random.randint(0, 1000) for _ in range(random.randint(0, 60))]
        build_max_heap(data)
        if not _is_max_heap(data):
            heap_ok = False
            break
    print(f"  {'build_max_heap property':22} {'PASS' if heap_ok else 'FAIL'}")

    ok_all = ok_all and heap_ok
    print("-" * 56)
    print("all passed" if ok_all else "SOME CASES FAILED")
    return ok_all


if __name__ == "__main__":
    random.seed(42)
    _run_checks()
