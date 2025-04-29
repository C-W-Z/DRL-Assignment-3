import math
import os
import random
import pickle
from collections import deque
from typing import Dict, List, Tuple

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

# 環境預處理 Wrapper
class SkipAndMax(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipAndMax, self).__init__(env)
        self.obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self.obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self.obs_buffer), axis=0)
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

    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_life = 2

    def reset(self):
        self._current_life = 2
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        life = info["life"]

        if reward <= 0:
            reward -= 0.1

        if life < self._current_life:
            reward -= 50

        self._current_life = life

        return state, reward, done, info

def make_env(env):
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipAndMax(env, skip=4)
    env = Frame_Processing(env)
    env = BufferingWrapper(env, n_steps=4)
    env = CustomReward(env)
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
        transition = super().store(obs, act, rew, next_obs, done)
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        return transition

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
            priority_alpha = priority ** self.alpha
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
    def __init__(self, in_channels: int, out_dim: int):
        super(Network, self).__init__()
        self.out_dim = out_dim

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
        self.advantage_layer = NoisyLinear(512, out_dim)
        self.value_hidden_layer = NoisyLinear(512, 512)
        self.value_layer = NoisyLinear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        advantage = self.advantage_layer(adv_hid)
        value = self.value_layer(val_hid)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

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

        self.dqn = Network(obs_shape[0], action_dim).to(self.device)
        self.dqn_target = Network(obs_shape[0], action_dim).to(self.device)
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
        self.best_avg_reward = -np.inf

    def select_action(self, state: np.ndarray) -> int:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_value = self.dqn(state)
            action = q_value.argmax().item()
        return action

    def step(self, action: int) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done

    def compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"]).to(self.device)
        done = torch.FloatTensor(samples["done"]).to(self.device)
        weights = torch.FloatTensor(samples["weights"]).to(self.device)

        # current Q value
        q_values = self.dqn(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)  # shape: (batch,)

        with torch.no_grad():
            # Double DQN: action from online, Q-value from target
            next_q_values = self.dqn(next_state)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)  # shape: (batch, 1)

            next_q_target = self.dqn_target(next_state)
            next_q_value = next_q_target.gather(1, next_actions).squeeze(1)

            target = reward + (1.0 - done) * self.gamma * next_q_value

        # TD Error
        td_error = q_value - target

        # Choose loss: MSE or Huber
        # elementwise_loss = F.mse_loss(q_value, target, reduction="none")
        elementwise_loss = F.smooth_l1_loss(q_value, target, reduction="none")

        # PER importance-sampling weighted loss
        loss = (elementwise_loss * weights).mean()
        return loss, td_error

    def update_model(self) -> torch.Tensor:
        if self.update_count % self.noise_reset_interval == 0:
            self.dqn.reset_noise()

        samples = self.memory.sample_batch(self.beta)
        loss, elementwise_loss = self.compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 5.0)
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
            'best_avg_reward': self.best_avg_reward,
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
        self.best_avg_reward = metadata['best_avg_reward']
        self.memory.set_state(metadata['memory_state'])

        print(f"Model loaded from {path}, metadata loaded from {metadata_path}")

    def train(self, num_episodes: int, save_interval: int = 100, plot_interval: int = 10):
        self.dqn.train()
        frame_idx = len(self.losses) + 1

        for _ in range(num_episodes):
            self.episode += 1

            episode_reward = 0
            state = self.env.reset()
            state = np.asarray(state)
            done = False

            while not done:
                self.total_steps += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                next_state = np.asarray(next_state)

                self.memory.store(state, action, reward, next_state, done)
                episode_reward += reward
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

            self.rewards.append(episode_reward)
            print(f"Episode {self.episode} | Frame {frame_idx} | Reward {episode_reward:.1f}")

            # if self.episode >= self.avg_window_size:
            #     avg_reward = np.mean(self.rewards[-self.avg_window_size:])
            #     if avg_reward > self.best_avg_reward:
            #         self.best_avg_reward = avg_reward
            #         self.save_model(f"{self.model_save_dir}/Best{self.best_avg_reward:.0f}_Episode{self.episode}.pth")

            if self.episode % save_interval == 0:
                self.save_model(f"{self.model_save_dir}/Episode{self.episode}.pth")

            if self.episode % plot_interval == 0:
                self.plot(self.episode)

        print(f"Best avg reward: {self.best_avg_reward}")

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
        memory_size=30000,
        batch_size=128,
        target_update=5000,
        seed=-1,
        gamma=0.99,
        alpha=0.6,
        beta=0.4,
        prior_eps=1e-6,
        n_step=5,
        tau=0.5,
        lr=0.00005,
        avg_window_size=100,
        model_save_dir="./models",
        plot_dir="./plots"
    )

    # agent.load_model("./models/Episode450.pth")
    agent.train(num_episodes=1000, save_interval=50, plot_interval=10)
    env.close()
