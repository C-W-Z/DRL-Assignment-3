from typing import Dict, Tuple, Deque
import numpy as np
from collections import deque
from segment_tree import SumSegmentTree, MinSegmentTree

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self,
        obs_shape: Tuple,
        size: int,
        batch_size: int = 32,
        n_step: int = 1,
        gamma: float = 0.99
    ):
        # self.obs_shape    = obs_shape
        self.obs_buf      = np.zeros((size,) + obs_shape, dtype=np.float32)
        self.next_obs_buf = np.zeros((size,) + obs_shape, dtype=np.float32)
        self.acts_buf     = np.zeros((size,), dtype=np.int32)
        self.rews_buf     = np.zeros((size,), dtype=np.float16)
        self.done_buf     = np.zeros((size,), dtype=np.bool8)
        self.max_size, self.batch_size = size, batch_size
        self.ptr  = 0
        self.size = 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        self.n_step_buffer.append((obs, act, rew, next_obs, done))

        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=np.take(self.obs_buf, idxs, axis=0),
            next_obs=np.take(self.next_obs_buf, idxs, axis=0),
            acts=np.take(self.acts_buf, idxs, axis=0),
            rews=np.take(self.rews_buf, idxs, axis=0).astype(np.float32),
            done=np.take(self.done_buf, idxs, axis=0).astype(np.float32),
            indices=idxs,
        )

    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[float, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in list(n_step_buffer)[-2::-1]:  # Iterate in reverse, excluding last
            r, n_o, d = transition[-3:]
            rew = r + gamma * rew * (1 - d)
            if d:
                next_obs, done = n_o, d

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
        self,
        obs_shape: Tuple,
        size: int,
        batch_size: int = 32,
        alpha: float = 0.6,
        n_step: int = 1,
        gamma: float = 0.99,
        prior_eps: float = 1e-6,
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            obs_shape, size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.prior_eps = prior_eps

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices].astype(np.float32)
        done = self.done_buf[indices].astype(np.float32)
        weights = self._calculate_weight(indices, beta)

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def _sample_proportional(self) -> np.ndarray:
        """Sample indices based on proportions."""
        indices = np.zeros(self.batch_size, dtype=np.int32)
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        # Generate all upper bounds at once
        arrange = np.arange(self.batch_size)
        bounds = np.random.uniform(
            segment * arrange,
            segment * (arrange + 1)
        )

        for i, upperbound in enumerate(bounds):
            indices[i] = self.sum_tree.retrieve(upperbound)

        return indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        priorities = np.maximum(priorities, self.prior_eps)

        for idx, priority in zip(indices, priorities):
            # assert priority > 0
            assert 0 <= idx < len(self)
            priority_alpha = priority ** self.alpha
            self.sum_tree[idx] = priority_alpha
            self.min_tree[idx] = priority_alpha

        self.max_priority = max(self.max_priority, np.max(priority))

    def _calculate_weight(self, indices: np.ndarray, beta: float) -> np.ndarray:
        _sum = self.sum_tree.sum()

        # get max weight
        p_min = self.min_tree.min() / _sum
        max_weight = (p_min * len(self)) ** (-beta)

        p_samples = np.array([self.sum_tree[idx] for idx in indices]) / _sum
        weights = (p_samples * len(self)) ** (-beta) / max_weight

        return weights

    def get_state(self):
        return {
            'obs_buf'      : self.obs_buf,
            'next_obs_buf' : self.next_obs_buf,
            'acts_buf'     : self.acts_buf,
            'rews_buf'     : self.rews_buf,
            'done_buf'     : self.done_buf,
            'ptr'          : self.ptr,
            'size'         : self.size,
            'n_step_buffer': list(self.n_step_buffer),
            'sum_tree'     : self.sum_tree.tree,
            'min_tree'     : self.min_tree.tree,
            'max_priority' : self.max_priority,
        }

    def set_state(self, state):
        self.obs_buf       = state['obs_buf']
        self.next_obs_buf  = state['next_obs_buf']
        self.acts_buf      = state['acts_buf']
        self.rews_buf      = state['rews_buf']
        self.done_buf      = state['done_buf']
        self.ptr           = state['ptr']
        self.size          = state['size']
        self.n_step_buffer = deque(state['n_step_buffer'], maxlen=self.n_step)
        self.sum_tree.tree = state['sum_tree']
        self.min_tree.tree = state['min_tree']
        self.max_priority  = state['max_priority']
