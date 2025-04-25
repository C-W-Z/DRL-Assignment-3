import math
import os
import random
import pickle
from collections import deque
from typing import Dict, List, Tuple
from tqdm import tqdm
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
import numba
import cv2
# import collections

from segment_tree import SumSegmentTree, MinSegmentTree

# 設置隨機種子以確保可重現性
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EpisodicLifeEnv(gym.Wrapper):

    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game
        over. Done by DeepMind for the DQN and co. since it helps value
        estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few fr
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped._life
        return obs

# 環境預處理 Wrapper
class SkipAndMax(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipAndMax, self).__init__(env)
        self.obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        origin_reward = 0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            # origin_reward += info.get('reward', 0)
            self.obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        # info['reward'] = origin_reward
        return max_frame, total_reward, done, info

    def reset(self):
        self.obs_buffer.clear()
        obs = self.env.reset()
        self.obs_buffer.append(obs)
        return obs

class Frame_Processing(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(Frame_Processing, self).__init__(env)
        old_shape = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8).shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, obs):
        return Frame_Processing.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Wrong resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1]).astype(np.uint8)
        return np.moveaxis(x_t, 2, 0)

class BufferingWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferingWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return np.array(self.buffer).astype(np.float32) / 255.0

class CustomReward(gym.Wrapper):

    def __init__(self, env, reward_shaping=True):
        super(CustomReward, self).__init__(env)
        self.reward_shaping = reward_shaping
        self._current_score = 0
        self._current_life = 2
        self._current_time = 400
        self._max_x_pos = 40
        self._current_coins = 0
        self._current_status = 'small'

    def reset(self):
        self._current_score = 0
        self._current_life = 2
        self._current_time = 400
        self._max_x_pos = 40
        self._current_coins = 0
        self._current_status = 'small'
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # origin reward
        info['reward'] = reward

        # info: {'coins': 0, 'flag_get': False, 'life': 2, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40, 'y_pos': 79}

        score = info["score"]
        life = info["life"]
        x_pos = info["x_pos"]
        coins = info['coins']
        status = info['status']
        time = info['time']

        shaped_reward = 0

        # 前進獎勵: 根據 x_pos 的增量
        # if x_pos > self._max_x_pos:
        #     self._max_x_pos = x_pos
            # shaped_reward += (x_pos - self._max_x_pos) / 5
        # 時間流逝
        # elif reward <= 0 and time < self._current_time:
        if reward <= 0 and time < self._current_time:
            shaped_reward -= 0.1

        shaped_reward += (score - self._current_score) / 10

        # # 硬幣獎勵 (coin): 根據 coins 的增量
        # coin = (coins - self._current_coins) * 10  # 每收集 1 個硬幣獎勵 10
        # shaped_reward += coin

        # # 狀態獎勵 (status): 根據 Mario 狀態的變化
        # status_reward = 0
        # status_map = {'small': 0, 'tall': 1, 'fireball': 2}  # 定義狀態的價值
        # current_status_value = status_map.get(self._current_status, 0)
        # new_status_value = status_map.get(status, 0)
        # if new_status_value > current_status_value:  # 狀態提升（例如 small -> tall）
        #     status_reward = 25  # 狀態提升獎勵
        # elif new_status_value < current_status_value:  # 狀態下降（例如 tall -> small）
        #     status_reward = -10  # 狀態下降懲罰
        # shaped_reward += status_reward

        # if life == 255:
        #     life = -1
        # if life < self._current_life or done or (time == 0 and self._current_time > 0):
        #     if self._current_time != 0:
        #         print("Died", life, self._current_time, self._max_x_pos)
        #         shaped_reward -= 50
        #     self._max_x_pos = x_pos
        #     info['end'] = True
        #     # print(info)

        self._current_score = score
        self._current_life = life
        self._current_time = time
        self._current_coins = coins
        self._current_status = status

        if info["flag_get"]:
            shaped_reward += 500

        if not self.reward_shaping:
            return state, reward, done, info
        return state, reward + shaped_reward, done, info

def make_env(env):
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    # env = EpisodicLifeEnv(env)
    # env = CustomReward(env, reward_shaping=False)
    env = SkipAndMax(env, skip=4)
    env = Frame_Processing(env)
    env = BufferingWrapper(env, n_steps=4)
    return env

# 2. Replay Buffer（基礎 N-step Buffer，使用 numba 加速）
class ReplayBuffer:
    def __init__(
        self,
        obs_shape: tuple,
        size: int,
        batch_size: int = 32,
        n_step: int = 1,
        gamma: float = 0.99
    ):
        self.obs_buf = np.zeros([size] + list(obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros([size] + list(obs_shape), dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

        self.n_step_buffer = np.zeros((n_step, 5), dtype=object)
        self.n_step_buffer_idx = 0
        self.n_step_buffer_full = False
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
        self.n_step_buffer[self.n_step_buffer_idx] = [obs, act, rew, next_obs, done]
        self.n_step_buffer_idx = (self.n_step_buffer_idx + 1) % self.n_step
        if not self.n_step_buffer_full and self.n_step_buffer_idx == 0:
            self.n_step_buffer_full = True

        if not self.n_step_buffer_full:
            return ()

        rews = np.array([t[2] for t in self.n_step_buffer], dtype=np.float32)
        dones = np.array([t[4] for t in self.n_step_buffer], dtype=np.float32)
        rew, next_obs, done = self._get_n_step_info(rews, dones, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4], self.gamma)
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return (obs, act, rew, next_obs, done)

    @staticmethod
    @numba.jit(nopython=True)
    def _get_n_step_info(rews: np.ndarray, dones: np.ndarray, last_next_obs: np.ndarray, last_done: float, gamma: float) -> Tuple[float, np.ndarray, float]:
        rew = rews[-1]
        next_obs = last_next_obs
        done = last_done
        for i in range(len(rews) - 2, -1, -1):
            r = rews[i]
            d = dones[i]
            rew = r + gamma * rew * (1 - d)
            if d:
                next_obs = last_next_obs
                done = d
        return rew, next_obs, done

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            indices=idxs,
        )

    def sample_batch_from_idxs(self, idxs: np.ndarray) -> Dict[str, np.ndarray]:
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size

# 3. Prioritized Replay Buffer（修改後適配無批次操作的 segment_tree）
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        obs_shape: tuple,
        size: int,
        batch_size: int = 32,
        alpha: float = 0.6,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        assert alpha >= 0
        super(PrioritizedReplayBuffer, self).__init__(obs_shape, size, batch_size, n_step, gamma)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # 增加 episode 成功度的追蹤
        self.episode_success = np.zeros(size, dtype=np.float32)  # 儲存每個 state 的 episode 成功度
        self.state_to_episode = np.zeros(size, dtype=np.int32)  # 每個 state 對應的 episode ID

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
        episode_id: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = super().store(obs, act, rew, next_obs, done)
        if transition:
            # 記錄 state 對應的 episode ID
            self.state_to_episode[self.tree_ptr] = episode_id

            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        return transition

    def update_episode_success(self, episode_id: int, success_score: float):
        """更新 episode 的成功度"""
        # 將成功度分配給該 episode 的所有 state
        for idx in range(self.max_size):
            if self.state_to_episode[idx] == episode_id:
                self.episode_success[idx] = success_score

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        assert len(self) >= self.batch_size, f"Buffer size {len(self)} is less than batch size {self.batch_size}"
        assert beta > 0

        p_total = self.sum_tree.sum(0, len(self) - 1)
        p_min = self.min_tree.min() / p_total if p_total > 0 else 1.0
        max_weight = (p_min * len(self)) ** (-beta) if p_min > 0 else 1.0

        indices = []
        p_samples = []
        for _ in range(self.batch_size):
            mass = np.random.uniform(0, p_total)
            idx = self.sum_tree.retrieve(mass)
            indices.append(idx)
            p_sample = self.sum_tree[idx] / p_total if p_total > 0 else 1.0
            p_samples.append(p_sample)

        indices = np.array(indices, dtype=np.int32)
        p_samples = np.array(p_samples, dtype=np.float32)
        weights = (p_samples * len(self)) ** (-beta)
        weights = np.where(max_weight > 0, weights / max_weight, 1.0)

        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        assert len(indices) == len(priorities), f"Indices length {len(indices)} does not match priorities length {len(priorities)}"
        priorities = np.maximum(priorities, 0)

        for idx, priority in zip(indices, priorities):
            assert 0 <= idx < len(self), f"Index {idx} out of bounds [0, {len(self)})"
            # 結合 episode 成功度調整優先級
            success_weight = self.episode_success[idx] if self.episode_success[idx] > 0 else 1.0
            adjusted_priority = priority * success_weight  # 乘以成功度權重
            priority_alpha = adjusted_priority ** self.alpha
            # priority_alpha = priority ** self.alpha
            self.sum_tree[idx] = priority_alpha
            self.min_tree[idx] = priority_alpha
            self.max_priority = max(self.max_priority, priority)

    def get_state(self) -> Dict[str, np.ndarray]:
        """保存 ReplayBuffer 的狀態"""
        return {
            'obs_buf': self.obs_buf,
            'next_obs_buf': self.next_obs_buf,
            'acts_buf': self.acts_buf,
            'rews_buf': self.rews_buf,
            'done_buf': self.done_buf,
            'ptr': self.ptr,
            'size': self.size,
            'n_step_buffer': self.n_step_buffer,
            'n_step_buffer_idx': self.n_step_buffer_idx,
            'n_step_buffer_full': self.n_step_buffer_full,
            'sum_tree_tree': self.sum_tree.tree,
            'min_tree_tree': self.min_tree.tree,
            'tree_ptr': self.tree_ptr,
            'max_priority': self.max_priority,
            'episode_success': self.episode_success,
            'state_to_episode': self.state_to_episode,
        }

    def set_state(self, state: Dict[str, np.ndarray]):
        """載入 ReplayBuffer 的狀態"""
        self.obs_buf = state['obs_buf']
        self.next_obs_buf = state['next_obs_buf']
        self.acts_buf = state['acts_buf']
        self.rews_buf = state['rews_buf']
        self.done_buf = state['done_buf']
        self.ptr = state['ptr']
        self.size = state['size']
        self.n_step_buffer = state['n_step_buffer']
        self.n_step_buffer_idx = state['n_step_buffer_idx']
        self.n_step_buffer_full = state['n_step_buffer_full']
        self.sum_tree.tree = state['sum_tree_tree']
        self.min_tree.tree = state['min_tree_tree']
        self.tree_ptr = state['tree_ptr']
        self.max_priority = state['max_priority']
        # self.episode_success = state['episode_success']
        # self.state_to_episode = state['state_to_episode']

# 4. Noisy Linear Layer
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

# 5. Network（改為 CNN）
class Network(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, atom_size: int, support: torch.Tensor):
        super(Network, self).__init__()
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        self.advantage_hidden_layer = NoisyLinear(512, 512)
        self.advantage_layer = NoisyLinear(512, out_dim * atom_size)
        self.value_hidden_layer = NoisyLinear(512, 512)
        self.value_layer = NoisyLinear(512, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        advantage = self.advantage_layer(adv_hid).view(-1, self.out_dim, self.atom_size)
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)
        return dist

    def reset_noise(self):
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

# 6. Rainbow DQN Agent
class RainbowDQNAgent:
    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        seed: int = -1,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta: float = 0.4,
        prior_eps: float = 1e-6,
        v_min: float = -500.0,
        v_max: float = 5000.0,
        atom_size: int = 51,
        n_step: int = 3,
        tau: float = 0.9,
        lr: float = 0.0001,
        avg_window_size: int = 100,
        model_save_dir: str = "./models",
        plot_dir: str = "./plots",
    ):
        if seed >= 0:
            set_seed(seed)
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_shape = env.observation_space.shape  # (4, 84, 84)
        action_dim = env.action_space.n  # 12

        self.memory = PrioritizedReplayBuffer(
            obs_shape=obs_shape,
            size=memory_size,
            batch_size=batch_size,
            alpha=alpha,
            n_step=n_step,
            gamma=gamma,
        )
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.prior_eps = prior_eps
        self.tau = tau
        self.model_save_dir = model_save_dir
        self.plot_dir = plot_dir

        self.beta = beta
        self.beta_increment = (1.0 - beta) / 1000000

        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(v_min, v_max, atom_size).to(self.device)

        self.dqn = Network(obs_shape[0], action_dim, atom_size, self.support).to(self.device)
        self.dqn_target = Network(obs_shape[0], action_dim, atom_size, self.support).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.total_steps = 0
        self.losses = []
        self.rewards = []
        self.update_count = 0
        self.noise_reset_interval = 10

        self.episode = 0
        self.avg_window_size = avg_window_size
        # self.best_avg_reward = -np.inf

    def select_action(self, state: np.ndarray) -> int:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_value = self.dqn(state)
            action = q_value.argmax().item()
        return action

    def compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"]).to(self.device)
        done = torch.FloatTensor(samples["done"]).to(self.device)
        weights = torch.FloatTensor(samples["weights"]).to(self.device)

        curr_dist = self.dqn.dist(state)
        curr_dist = curr_dist[range(self.batch_size), action]

        with torch.no_grad():
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward.unsqueeze(1) + (1 - done).unsqueeze(1) * (self.gamma ** self.memory.n_step) * self.support.unsqueeze(0)
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / ((self.v_max - self.v_min) / (self.atom_size - 1))
            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size).long() \
                .unsqueeze(1).expand(self.batch_size, self.atom_size).to(self.device)

            proj_dist = torch.zeros_like(next_dist)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        log_p = torch.log(curr_dist.clamp(min=1e-3))
        elementwise_loss = -(proj_dist * log_p).sum(1)
        loss = (weights * elementwise_loss).mean()
        return loss, elementwise_loss

    def update_model(self) -> torch.Tensor:
        if self.update_count % self.noise_reset_interval == 0:
            self.dqn.reset_noise()

        samples = self.memory.sample_batch(self.beta)
        loss, elementwise_loss = self.compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        priorities = elementwise_loss.abs().detach().cpu().numpy() + self.prior_eps
        self.memory.update_priorities(samples["indices"], priorities)

        self.beta = min(1.0, self.beta + self.beta_increment)
        return loss.item()

    def target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def target_soft_update(self):
        for target_param, param in zip(self.dqn_target.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, path: str):
        # 保存權重和優化器狀態
        torch.save(self.dqn.state_dict(), path)

        # 保存其他數據到單獨的文件
        metadata_path = path.replace('.pth', '_metadata.pkl')
        metadata = {
            'dqn_target_state_dict': self.dqn_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rewards': self.rewards,
            'losses': self.losses,
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'beta': self.beta,
            'episode': self.episode,
            # 'best_avg_reward': self.best_avg_reward,
            'memory_state': self.memory.get_state(),
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Model saved to {path}, metadata saved to {metadata_path}")

    def load_model(self, path, eval_mode=False):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found")

        # 載入權重
        self.dqn.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        if eval_mode:
            self.dqn.eval()
            print(f"Model loaded from {path}")
            return

        # 載入其他數據
        metadata_path = path.replace('.pth', '_metadata.pkl')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file {metadata_path} not found")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        self.dqn_target.load_state_dict(metadata['dqn_target_state_dict'])
        self.optimizer.load_state_dict(metadata['optimizer_state_dict'])
        self.rewards = metadata['rewards']
        self.losses = metadata['losses']
        self.total_steps = metadata['total_steps']
        self.update_count = metadata['update_count']
        self.beta = metadata['beta']
        self.episode = metadata['episode']
        # self.best_avg_reward = metadata['best_avg_reward']
        self.memory.set_state(metadata['memory_state'])

        print(f"Model loaded from {path}, metadata loaded from {metadata_path}")

    def train(self, num_episodes: int, save_interval: int = 100, plot_interval: int = 10):
        self.dqn.train()
        frame_idx = len(self.losses) + 1

        for _ in range(num_episodes):
            self.episode += 1

            episode_reward = 0
            # episode_reward_origin = 0

            state = self.env.reset()
            state = np.asarray(state)
            done = False

            # cur_life = 2

            while not done:
                self.total_steps += 1
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.asarray(next_state)
                # self.env.render()

                self.memory.store(state, action, reward, next_state, done, self.episode)
                episode_reward += reward
                # episode_reward_origin += info['reward']

                # Episode End
                # new_life = info['life']
                # if new_life == 255:
                #     new_life = -1
                # if new_life < cur_life or done or info.get('end', False):

                #     # 當前生命週期結束，計算成功度並更新
                #     success_score = episode_reward / 1000.0  # 標準化成功度
                #     # if info['x_pos'] >= 1320 and episode_reward < 1000 and new_life != 1:
                #     #     success_score += self.rewards[-1] / 1000.0
                #     if info['flag_get']:
                #         success_score += 5.0  # 通關給予額外權重
                #     success_score = max(success_score, self.prior_eps)
                #     self.memory.update_episode_success(self.episode, success_score)

                #     self.rewards.append(episode_reward_origin)
                #     print(f"Episode {self.episode} | Frame {frame_idx} | Reward {episode_reward:.1f} | Origin Reward {episode_reward_origin:.1f}")

                #     if self.episode % save_interval == 0:
                #         self.save_model(f"{self.model_save_dir}/Episode{self.episode}.pth")

                #     if self.episode % plot_interval == 0:
                #         self.plot(self.episode)

                #     if not done:
                #         cur_life = new_life
                #         episode_reward = 0
                #         episode_reward_origin = 0
                #         self.episode += 1

                state = next_state

                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()
                    self.losses.append(loss)
                    self.update_count += 1

                    if self.update_count % self.target_update == 0:
                        if self.tau == 1:
                            self.target_hard_update()
                        else:
                            self.target_soft_update()

                frame_idx += 1

            # 當前生命週期結束，計算成功度並更新
            # success_score = episode_reward / 3000.0  # 標準化成功度
            # if info['x_pos'] >= 1320 and episode_reward < 1000 and new_life != 1:
            #     success_score += self.rewards[-1] / 1000.0
            # if info['flag_get']:
            #     success_score += 5.0  # 通關給予額外權重
            # if episode_reward > 4000:
            #     success_score += 1.0
            # success_score = max(success_score, self.prior_eps)
            # self.memory.update_episode_success(self.episode, success_score)

            self.rewards.append(episode_reward)
            print(f"Episode {self.episode} | Frame {frame_idx} | Reward {episode_reward:.0f} | Stage {info['stage']}")

            if self.episode % save_interval == 0:
                self.save_model(f"{self.model_save_dir}/Episode{self.episode}.pth")

            if self.episode % plot_interval == 0:
                self.plot(self.episode)

            # if self.episode >= self.avg_window_size:
            #     avg_reward = np.mean(self.rewards[-self.avg_window_size:])
            #     if avg_reward > self.best_avg_reward:
            #         self.best_avg_reward = avg_reward
            #         self.save_model(f"{self.model_save_dir}/Best{self.best_avg_reward:.0f}_Episode{self.episode}.pth")

        # print(f"Best avg reward: {self.best_avg_reward}")

    def plot(self, episode: int):
        # clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(121)
        if self.episode >= self.avg_window_size:
            avg_reward = np.mean(self.rewards[-self.avg_window_size:])
        else:
            avg_reward = np.mean(self.rewards)
        plt.title(f"Episode {episode} | Avg Reward {avg_reward:.2f}")
        plt.plot(self.rewards)
        plt.subplot(122)
        plt.title("Loss")
        plt.plot(self.losses)
        # plt.show()

        save_path = os.path.join(self.plot_dir, f"episode_{episode}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # 關閉圖表，避免記憶體洩漏
        print(f"Plot saved to {save_path}")

# 主程式
if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = make_env(env)
    # print(env.observation_space.shape)  # (4, 84, 84)
    # print(env.action_space.n)           # 12

    agent = RainbowDQNAgent(
        env=env,
        memory_size=50000,
        batch_size=128,
        target_update=5000,
        seed=-1,
        gamma=0.99,
        alpha=0.6,
        beta=0.4,
        prior_eps=1e-6,
        v_min=-1000.0,
        v_max=7000.0,
        atom_size=51,
        n_step=5,
        tau=0.5,
        lr=0.000005,
        avg_window_size=100,
        model_save_dir="./models",
        plot_dir="./plots"
    )

    agent.load_model("./models/Episode580.pth", eval_mode=False)
    # agent.beta = 0.5

    agent.dqn.train()
    frame_idx = len(agent.losses) + 1

    # ids = [1, 8, 9, 11, 13, 14, 19, 22, 24, 28, 30, 31, 32, 33, 34]
    ids = [40, 41, 42, 43, 44]

    for id in ids:
        agent.episode += 1

        path = f"./human_play/play_{id}.pkl"
        with open(path, 'rb') as f:
            play = pickle.load(f)
        episode_reward = play['total_reward']
        trajectory = play['trajectory']

        for state, action, reward, next_state, done in trajectory:
            agent.memory.store(state, action, reward, next_state, done, agent.episode)

            if len(agent.memory) >= agent.batch_size:
                loss = agent.update_model()
                agent.losses.append(loss)
                agent.update_count += 1

                if agent.update_count % agent.target_update == 0:
                    if agent.tau == 1:
                        agent.target_hard_update()
                    else:
                        agent.target_soft_update()

            frame_idx += 1

        # 當前生命週期結束，計算成功度並更新
        # success_score = episode_reward / 3000.0  # 標準化成功度
        # success_score = max(success_score, agent.prior_eps)
        # agent.memory.update_episode_success(agent.episode, success_score)

        agent.rewards.append(episode_reward)
        print(f"Episode {agent.episode} | Frame {frame_idx} | Reward {episode_reward:.0f}")

    # for _ in tqdm(range(2000)):
    #     if len(agent.memory) >= agent.batch_size:
    #         loss = agent.update_model()
    #         agent.losses.append(loss)
    #         agent.update_count += 1

    #         if agent.update_count % agent.target_update == 0:
    #             if agent.tau == 1:
    #                 agent.target_hard_update()
    #             else:
    #                 agent.target_soft_update()
    #     frame_idx += 1

    agent.save_model(f"{agent.model_save_dir}/Episode{agent.episode}.pth")
    agent.plot(agent.episode)

    agent.train(num_episodes=3000, save_interval=10, plot_interval=10)
    env.close()
