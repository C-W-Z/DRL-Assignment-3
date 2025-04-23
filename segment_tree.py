# -*- coding: utf-8 -*-
"""Segment tree for Prioritized Replay Buffer, optimized with Numba."""
"""
  Original code is from OpenAI baselines github repository:
  https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
"""

import numpy as np
import numba
from typing import Callable

@numba.jit(nopython=True)
def _operate_helper_sum(tree: np.ndarray, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
    """Helper function for sum operation, optimized with Numba."""
    if start == node_start and end == node_end:
        return tree[node]
    mid = (node_start + node_end) // 2
    if end <= mid:
        return _operate_helper_sum(tree, start, end, 2 * node, node_start, mid)
    else:
        if mid + 1 <= start:
            return _operate_helper_sum(tree, start, end, 2 * node + 1, mid + 1, node_end)
        else:
            left = _operate_helper_sum(tree, start, mid, 2 * node, node_start, mid)
            right = _operate_helper_sum(tree, mid + 1, end, 2 * node + 1, mid + 1, node_end)
            return left + right

@numba.jit(nopython=True)
def _operate_helper_min(tree: np.ndarray, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
    """Helper function for min operation, optimized with Numba."""
    if start == node_start and end == node_end:
        return tree[node]
    mid = (node_start + node_end) // 2
    if end <= mid:
        return _operate_helper_min(tree, start, end, 2 * node, node_start, mid)
    else:
        if mid + 1 <= start:
            return _operate_helper_min(tree, start, end, 2 * node + 1, mid + 1, node_end)
        else:
            left = _operate_helper_min(tree, start, mid, 2 * node, node_start, mid)
            right = _operate_helper_min(tree, mid + 1, end, 2 * node + 1, mid + 1, node_end)
            return min(left, right)

@numba.jit(nopython=True)
def _update_tree_sum(tree: np.ndarray, idx: int, val: float, capacity: int):
    """Update the tree for sum operation, optimized with Numba."""
    idx += capacity
    tree[idx] = val
    idx //= 2
    while idx >= 1:
        tree[idx] = tree[2 * idx] + tree[2 * idx + 1]
        idx //= 2

@numba.jit(nopython=True)
def _update_tree_min(tree: np.ndarray, idx: int, val: float, capacity: int):
    """Update the tree for min operation, optimized with Numba."""
    idx += capacity
    tree[idx] = val
    idx //= 2
    while idx >= 1:
        tree[idx] = min(tree[2 * idx], tree[2 * idx + 1])
        idx //= 2

@numba.jit(nopython=True)
def _retrieve_sum(tree: np.ndarray, upperbound: float, capacity: int) -> int:
    """Retrieve the index for sum tree, optimized with Numba."""
    idx = 1
    while idx < capacity:  # while non-leaf
        left = 2 * idx
        right = left + 1
        if tree[left] > upperbound:
            idx = 2 * idx
        else:
            upperbound -= tree[left]
            idx = right
    return idx - capacity

class SegmentTree:
    """Base SegmentTree class with Numba optimization."""
    def __init__(self, capacity: int, operation: Callable, init_value: float):
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = np.full(2 * capacity, init_value, dtype=np.float32)  # 使用 NumPy 陣列
        self.operation = operation

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1
        # 根據 operation 選擇對應的 Numba 函數
        if self.operation == "sum":
            return _operate_helper_sum(self.tree, start, end, 1, 0, self.capacity - 1)
        elif self.operation == "min":
            return _operate_helper_min(self.tree, start, end, 1, 0, self.capacity - 1)
        else:
            raise ValueError("Unsupported operation")

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        # 根據 operation 選擇對應的 Numba 函數
        if self.operation == "sum":
            _update_tree_sum(self.tree, idx, val, self.capacity)
        elif self.operation == "min":
            _update_tree_min(self.tree, idx, val, self.capacity)
        else:
            raise ValueError("Unsupported operation")

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity
        return self.tree[self.capacity + idx]

class SumSegmentTree(SegmentTree):
    """SumSegmentTree optimized with Numba."""
    def __init__(self, capacity: int):
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation="sum", init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return self.operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree."""
        total_sum = self.sum()
        assert 0 <= upperbound <= total_sum + 1e-5, f"upperbound: {upperbound}, total_sum: {total_sum}"
        return _retrieve_sum(self.tree, upperbound, self.capacity)

class MinSegmentTree(SegmentTree):
    """MinSegmentTree optimized with Numba."""
    def __init__(self, capacity: int):
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation="min", init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return self.operate(start, end)
