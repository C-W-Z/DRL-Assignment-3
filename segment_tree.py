# -*- coding: utf-8 -*-
"""
    Segment tree for Prioritized Replay Buffer, optimized with Numba.
    Original code is from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
"""

import numpy as np
from numba import njit
from typing import Tuple

@njit
def _operate_helper_sum(tree: np.ndarray, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
    """Helper function for sum operation, optimized with Numba."""
    if start == node_start and end == node_end:
        return tree[node]
    mid = (node_start + node_end) >> 1 # // 2
    if end <= mid:
        return _operate_helper_sum(tree, start, end, node << 1, node_start, mid) # 2 * node
    else:
        if mid + 1 <= start:
            return _operate_helper_sum(tree, start, end, (node << 1) + 1, mid + 1, node_end) # 2 * node + 1
        else:
            left = _operate_helper_sum(tree, start, mid, node << 1, node_start, mid) # 2 * node
            right = _operate_helper_sum(tree, mid + 1, end, (node << 1) + 1, mid + 1, node_end) # 2 * node + 1
            return left + right

@njit
def _operate_helper_min(tree: np.ndarray, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
    """Helper function for min operation, optimized with Numba."""
    if start == node_start and end == node_end:
        return tree[node]
    mid = (node_start + node_end) >> 1 # // 2
    if end <= mid:
        return _operate_helper_min(tree, start, end, node << 1, node_start, mid) # 2 * node
    else:
        if mid + 1 <= start:
            return _operate_helper_min(tree, start, end, (node << 1) + 1, mid + 1, node_end) # 2 * node + 1
        else:
            left = _operate_helper_min(tree, start, mid, node << 1, node_start, mid) # 2 * node
            right = _operate_helper_min(tree, mid + 1, end, (node << 1) + 1, mid + 1, node_end) # 2 * node + 1
            return min(left, right)

@njit
def _update_tree_sum(tree: np.ndarray, idx: int, val: float, capacity: int):
    """Update the tree for sum operation, optimized with Numba."""
    idx += capacity
    tree[idx] = val
    idx >>= 1 # //= 2
    while idx >= 1:
        tree[idx] = tree[idx << 1] + tree[(idx << 1) + 1] # 2 * idx, 2 * idx + 1
        idx >>= 1 # //= 2

@njit
def _update_tree_min(tree: np.ndarray, idx: int, val: float, capacity: int):
    """Update the tree for min operation, optimized with Numba."""
    idx += capacity
    tree[idx] = val
    idx >>= 1 # //= 2
    while idx >= 1:
        tree[idx] = min(tree[idx << 1], tree[(idx << 1) + 1]) # 2 * idx, 2 * idx + 1
        idx >>= 1 # //= 2

@njit
def _retrieve_sum(tree: np.ndarray, upperbound: float, capacity: int) -> int:
    """Retrieve the index for sum tree, optimized with Numba."""
    idx = 1
    while idx < capacity:  # while non-leaf
        left = idx << 1  # 2 * idx
        right = left + 1
        if tree[left] > upperbound:
            idx <<= 1 # *= 2
        else:
            upperbound -= tree[left]
            idx = right
    return idx - capacity

@njit
def _batch_update_tree_sum(tree: np.ndarray, indices: np.ndarray, values: np.ndarray, capacity: int):
    for i in range(len(indices)):
        idx = indices[i] + capacity
        tree[idx] = values[i]
        idx >>= 1 # //= 2
        while idx >= 1:
            tree[idx] = tree[idx << 1] + tree[(idx << 1) + 1] # 2 * idx, 2 * idx + 1
            idx >>= 1 # //= 2

@njit
def _batch_update_tree_min(tree: np.ndarray, indices: np.ndarray, values: np.ndarray, capacity: int):
    for i in range(len(indices)):
        idx = indices[i] + capacity
        tree[idx] = values[i]
        idx >>= 1 # //= 2
        while idx >= 1:
            tree[idx] = min(tree[idx << 1], tree[(idx << 1) + 1]) # 2 * idx, 2 * idx + 1
            idx >>= 1 # //= 2

@njit
def _sample_core(
    sum_tree: np.ndarray,
    min_tree: np.ndarray,
    size: int,
    batch_size: int,
    beta: float,
    capacity: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core sampling logic for PrioritizedReplayBuffer, optimized with Numba.

    Args:
        sum_tree: SumSegmentTree 的底層陣列 (self.sum_tree.tree)
        min_tree: MinSegmentTree 的底層陣列 (self.min_tree.tree)
        size: 當前緩衝區大小
        batch_size: 批次大小
        frame_idx: 當前幀索引（用於計算 beta）
        capacity: 線段樹的容量（葉節點數量）

    Returns:
        indices: 採樣的索引陣列
        weights: 計算得到的權重陣列
    """
    # 計算 p_total 和 p_min
    p_total = _operate_helper_sum(sum_tree, 0, size - 1, 1, 0, capacity - 1)
    p_min = _operate_helper_min(min_tree, 0, size - 1, 1, 0, capacity - 1) / p_total if p_total > 0 else 1.0

    # 計算 max_weight
    max_weight = (p_min * size) ** (-beta) if p_min > 0 else 1.0

    # 生成隨機數 masses
    masses = np.random.uniform(0, p_total, size=batch_size)

    # 預分配 indices 和 p_samples 陣列
    indices = np.zeros(batch_size, dtype=np.int32)
    p_samples = np.zeros(batch_size, dtype=np.float32)

    # 逐個處理 masses，檢索索引並計算 p_samples
    for i in range(batch_size):
        indices[i] = _retrieve_sum(sum_tree, masses[i], capacity)
        p_samples[i] = sum_tree[capacity + indices[i]] / p_total

    # 計算 weights
    weights = ((p_samples * size) ** (-beta)) / max_weight

    return indices, weights

class SegmentTree:
    """Base SegmentTree class with Numba optimization."""
    def __init__(self, capacity: int, operation: str, init_value: float):
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

    def batch_update(self, indices: np.ndarray, values: np.ndarray):
        return _batch_update_tree_sum(self.tree, indices, values, self.capacity)

class MinSegmentTree(SegmentTree):
    """MinSegmentTree optimized with Numba."""
    def __init__(self, capacity: int):
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation="min", init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return self.operate(start, end)

    def batch_update(self, indices: np.ndarray, values: np.ndarray):
        return _batch_update_tree_min(self.tree, indices, values, self.capacity)
