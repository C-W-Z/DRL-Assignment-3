import os
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple
import cv2
import gym
from gym.wrappers import TimeLimit
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import transforms as T
from tqdm import tqdm
from numba import njit

from segment_tree import SumSegmentTree, MinSegmentTree

Exp = namedtuple('Exp', ['state', 'action', 'reward', 'next_state', 'done'])

# -----------------------------
# Hyperparameters
# -----------------------------
MEMORY_SIZE             = 50000
BATCH_SIZE              = 64
GAMMA                   = 0.95
TARGET_UPDATE           = 10000
NOISY_STD_INIT          = 2.5
LR                      = 0.00025
ADAM_EPS                = 0.00015
V_MIN                   = -1000.0
V_MAX                   = 10000.0
ATOM_SIZE               = 51
N_STEP                  = 5
ALPHA                   = 0.6
BETA_START              = 0.4
BETA_FRAMES             = 1000000
PRIOR_EPS               = 1e-6
TAU                     = 0.95
SKIP_FRAMES             = 4
STACK_FRAMES            = 4
MAX_EPISODE_STEPS       = 3000
MAX_FRAMES              = 1000000
BACKWARD_PENALTY        = 0
STAY_PENALTY            = 0
DEATH_PENALTY           = -100
ICM_BETA                = 0.2
ICM_ETA                 = 0.05
ICM_LR                  = 1e-4
ICM_EMBED_DIM           = 256
EVAL_INTERVAL           = 10
SAVE_INTERVAL           = 100
PLOT_INTERVAL           = 10
MODEL_DIR               = "./models"
PLOT_DIR                = "./plots"

GAMMA_POW_N_STEP = GAMMA ** N_STEP

@njit
def get_beta_by_frame(frame_idx):
    return min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

# -----------------------------
# 1. Environment Wrappers
# -----------------------------
class SkipAndMax(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer[i & 1] = np.asarray(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(self._obs_buffer, axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = np.asarray(obs)
        self._obs_buffer[0] = self._obs_buffer[1] = obs
        return obs

class FrameProcessing(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # old_shape = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8).shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, 84, 84), dtype=np.float32)

    def observation(self, obs):
        return FrameProcessing.process(obs)

    @staticmethod
    def process(frame):
        frame = np.asarray(frame)
        # assert frame.shape == (240, 256, 3), "Wrong resolution."
        # 使用 OpenCV 轉灰度
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # (240, 256), uint8
        # 縮放到 (110, 84)，然後裁剪
        frame = cv2.resize(frame, (84, 110), interpolation=cv2.INTER_AREA)
        frame = frame[18:102, :]  # (84, 84)
        # 轉為 (1, 84, 84)，規範化到 [0.0, 1.0]
        return frame.astype(np.float32)[np.newaxis, :, :] / 255.0
        # assert frame.shape == (1, 84, 84)

class FrameStack(gym.Wrapper):
    def __init__(self, env, n_steps=4):
        super().__init__(env)
        self.n_steps = n_steps
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 1, shape=(shp[0] * n_steps, shp[1], shp[2]), dtype=np.float32)
        self.frames = np.zeros(self.observation_space.shape, dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        obs = np.asarray(obs)
        for i in range(self.n_steps):
            self.frames[i] = obs
        return self.frames

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.asarray(obs)
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = obs
        return self.frames, reward, done, info

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipAndMax(env, skip=SKIP_FRAMES)
    env = FrameProcessing(env)
    env = FrameStack(env, n_steps=STACK_FRAMES)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    return env

# -----------------------------
# 2. Noisy Linear Layer
# -----------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init=NOISY_STD_INIT):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon
            )
        return F.linear(x, self.weight_mu, self.bias_mu)

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

# -----------------------------
# 3. Intrinsic Curiosity Module
# -----------------------------
class ICM(nn.Module):
    def __init__(self, feat_dim, n_actions, embed_dim=ICM_EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(embed_dim + n_actions, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, feat, next_feat, action):
        phi = self.encoder(feat)
        phi_next = self.encoder(next_feat)

        # Normalize embeddings to unit L2 norm
        phi = F.normalize(phi, p=2, dim=1)
        phi_next = F.normalize(phi_next, p=2, dim=1)

        inv_in = torch.cat([phi, phi_next], dim=1)
        logits = self.inverse_model(inv_in)
        a_onehot = F.one_hot(action, logits.size(-1)).float()
        fwd_in = torch.cat([phi, a_onehot], dim=1)
        pred_phi_next = self.forward_model(fwd_in)
        return logits, pred_phi_next, phi_next

# -----------------------------
# 4. Dueling Distributional Network
# -----------------------------
class DuelingDistNetwork(nn.Module):
    def __init__(self, in_channels: int, n_actions: int, atom_size: int, support: torch.Tensor):
        super().__init__()
        self.support = support
        self.n_actions = n_actions
        self.atom_size = atom_size

        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            self.feat_dim = self.feature_layer(dummy).shape[1]

        self.advantage_hidden_layer = NoisyLinear(self.feat_dim, 512)
        self.advantage_layer = NoisyLinear(512, n_actions * atom_size)
        self.value_hidden_layer = NoisyLinear(self.feat_dim, 512)
        self.value_layer = NoisyLinear(512, atom_size)

    def get_features(self, x):
        return self.feature_layer(x / 255.0)

    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x):
        x = self.get_features(x)

        advantage = F.relu(self.advantage_hidden_layer(x))
        advantage = self.advantage_layer(advantage).view(-1, self.n_actions, self.atom_size)

        value = F.relu(self.value_hidden_layer(x))
        value = self.value_layer(value).view(-1, 1, self.atom_size)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)
        return dist

    def reset_noise(self):
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

# -----------------------------
# 5. Prioritized Replay Buffer
# -----------------------------
@njit
def _compute_n_step_return(
    rewards: np.ndarray,
    dones: np.ndarray,
    last_next_state: np.ndarray,
    last_done: float,
    gamma: float
) -> Tuple[float, np.ndarray, float]:
    reward = rewards[-1]
    next_state = last_next_state
    done = last_done
    for i in range(len(rewards) - 2, -1, -1):
        r = rewards[i]
        d = dones[i]
        reward = r + gamma * reward * (1 - d)
        if d:
            next_state = last_next_state
            done = d
    return reward, next_state, done

@njit
def _compute_weights(p_samples: np.ndarray, size: int, beta: float, max_weight: float) -> np.ndarray:
    weights = (p_samples * size) ** (-beta)
    return weights / max_weight

class PrioritizedReplayBuffer:
    def __init__(self, obs_shape, size, batch_size):
        self.obs_shape = obs_shape
        self.max_size, self.batch_size = size, batch_size
        self.obs_buf = np.zeros([size] + list(obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros([size] + list(obs_shape), dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.int64)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.ptr, self.size = 0, 0
        self.n_step_buffer = deque(maxlen=N_STEP)
        self.max_priority, self.tree_ptr = 1.0, 0
        tree_capacity = 1
        while tree_capacity < size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, state, action, reward, next_state, done):
        self.n_step_buffer.append(Exp(state, action, reward, next_state, done))
        if len(self.n_step_buffer) < N_STEP:
            return
        reward_n, next_state_n, done_n = self._get_n_step()
        self.obs_buf[self.ptr] = self.n_step_buffer[0].state
        self.acts_buf[self.ptr] = self.n_step_buffer[0].action
        self.rews_buf[self.ptr] = reward_n
        self.next_obs_buf[self.ptr] = next_state_n
        self.done_buf[self.ptr] = done_n
        self.sum_tree[self.tree_ptr] = self.max_priority ** ALPHA
        self.min_tree[self.tree_ptr] = self.max_priority ** ALPHA
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _get_n_step(self):
        # 提取 n_step_buffer 的資料為 NumPy 陣列
        rewards = np.array([trans.reward for trans in self.n_step_buffer], dtype=np.float32)
        dones = np.array([trans.done for trans in self.n_step_buffer], dtype=np.float32)
        last_next_state = self.n_step_buffer[-1].next_state
        last_done = float(self.n_step_buffer[-1].done)
        # 用 numba 加速計算
        reward, next_state, done = _compute_n_step_return(rewards, dones, last_next_state, last_done, GAMMA)
        return reward, next_state, done

    def sample(self, frame_idx):
        if self.size < self.batch_size:
            return None
        p_total = self.sum_tree.sum(0, self.size - 1)
        p_min = self.min_tree.min() / p_total if p_total > 0 else 1.0
        beta = get_beta_by_frame(frame_idx)
        max_weight = (p_min * self.size) ** (-beta) if p_min > 0 else 1.0
        indices = []
        p_samples = []
        for _ in range(self.batch_size):
            mass = random.uniform(0, p_total)
            idx = self.sum_tree.retrieve(mass)
            indices.append(idx)
            p_sample = self.sum_tree[idx] / p_total
            p_samples.append(p_sample)
        indices = np.array(indices)
        p_samples = np.array(p_samples)
        weights = _compute_weights(p_samples, self.size, beta, max_weight)
        batch = Exp(
            state=self.obs_buf[indices],
            action=self.acts_buf[indices],
            reward=self.rews_buf[indices],
            next_state=self.next_obs_buf[indices],
            done=self.done_buf[indices]
        )
        return batch, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = np.maximum(priorities, PRIOR_EPS)
        priorities_alpha = priorities ** ALPHA
        indices = np.asarray(indices, dtype=np.int32)
        # 直接調用 segment_tree 中的 batch_update 方法
        self.sum_tree.batch_update(indices, priorities_alpha)
        self.min_tree.batch_update(indices, priorities_alpha)
        self.max_priority = max(self.max_priority, np.max(priorities))

    def get_state(self):
        return {
            'obs_buf': self.obs_buf,
            'next_obs_buf': self.next_obs_buf,
            'acts_buf': self.acts_buf,
            'rews_buf': self.rews_buf,
            'done_buf': self.done_buf,
            'ptr': self.ptr,
            'size': self.size,
            'n_step_buffer': list(self.n_step_buffer),
            'sum_tree': self.sum_tree.tree,
            'min_tree': self.min_tree.tree,
            'tree_ptr': self.tree_ptr,
            'max_priority': self.max_priority,
        }

    def set_state(self, state):
        self.obs_buf = state['obs_buf']
        self.next_obs_buf = state['next_obs_buf']
        self.acts_buf = state['acts_buf']
        self.rews_buf = state['rews_buf']
        self.done_buf = state['done_buf']
        self.ptr = state['ptr']
        self.size = state['size']
        self.n_step_buffer = deque(state['n_step_buffer'], maxlen=N_STEP)
        self.sum_tree.tree = state['sum_tree']
        self.min_tree.tree = state['min_tree']
        self.tree_ptr = state['tree_ptr']
        self.max_priority = state['max_priority']

# -----------------------------
# 6. Agent
# -----------------------------
class Agent:
    def __init__(self, obs_shape: Tuple, n_actions: int, device=None):
        self.device = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions = n_actions

        self.support = torch.linspace(V_MIN, V_MAX, ATOM_SIZE).to(self.device)

        self.online = DuelingDistNetwork(obs_shape[0], n_actions, ATOM_SIZE, self.support).to(self.device)

        self.target = DuelingDistNetwork(obs_shape[0], n_actions, ATOM_SIZE, self.support).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=LR, eps=ADAM_EPS)

        self.icm = ICM(self.online.feat_dim, n_actions).to(self.device)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=ICM_LR)

        self.buffer = PrioritizedReplayBuffer(obs_shape, MEMORY_SIZE, BATCH_SIZE)

        self.frame_idx = 0
        self.rewards = []
        self.dqn_losses = []
        self.icm_losses = []
        self.eval_rewards = []
        self.int_rewards = []
        self.best_eval_reward = -np.inf

    def act(self, state):
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online(s_t)
        return int(q.argmax(1).item())

    def learn(self):
        sample = self.buffer.sample(self.frame_idx)
        if sample is None:
            return None, None, None
        batch, weights, indices = sample
        state = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        action = torch.tensor(batch.action, dtype=torch.int64, device=self.device)
        r_ext = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float32, device=self.device)
        w = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Compute distributional loss
        curr_dist = self.online.dist(state)[range(BATCH_SIZE), action]
        with torch.no_grad():
            next_action = self.online(next_state).argmax(1)
            next_dist = self.target.dist(next_state)[range(BATCH_SIZE), next_action]
            t_z = r_ext.unsqueeze(1) + (1 - done).unsqueeze(1) * GAMMA_POW_N_STEP * self.support.unsqueeze(0)
            t_z = t_z.clamp(min=V_MIN, max=V_MAX)
            b = (t_z - V_MIN) / ((V_MAX - V_MIN) / (ATOM_SIZE - 1))
            l = b.floor().long()
            u = b.ceil().long()
            offset = torch.linspace(0, (BATCH_SIZE - 1) * ATOM_SIZE, BATCH_SIZE).long().unsqueeze(1).expand(BATCH_SIZE, ATOM_SIZE).to(self.device)
            proj_dist = torch.zeros_like(next_dist)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        log_p = torch.log(curr_dist.clamp(min=1e-3))
        elementwise_loss = -(proj_dist * log_p).sum(1)
        dqn_loss = (elementwise_loss * w).mean()

        # ICM loss
        feat = self.online.get_features(state)
        next_feat = self.online.get_features(next_state)
        logits, pred_phi_n, true_phi_n = self.icm(feat.detach(), next_feat.detach(), action)
        inv_loss = F.cross_entropy(logits, action)
        fwd_loss = F.mse_loss(pred_phi_n, true_phi_n.detach())
        icm_loss = (1 - ICM_BETA) * inv_loss + ICM_BETA * fwd_loss
        with torch.no_grad():
            int_r = ICM_ETA * 0.5 * (pred_phi_n - true_phi_n).pow(2).sum(dim=1)

        # Update DQN with combined reward
        total_r = r_ext + int_r
        curr_dist = self.online.dist(state)[range(BATCH_SIZE), action]
        with torch.no_grad():
            t_z = total_r.unsqueeze(1) + (1 - done).unsqueeze(1) * GAMMA_POW_N_STEP * self.support.unsqueeze(0)
            t_z = t_z.clamp(min=V_MIN, max=V_MAX)
            b = (t_z - V_MIN) / ((V_MAX - V_MIN) / (ATOM_SIZE - 1))
            l = b.floor().long()
            u = b.ceil().long()
            proj_dist = torch.zeros_like(next_dist)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        log_p = torch.log(curr_dist.clamp(min=1e-3))
        elementwise_loss = -(proj_dist * log_p).sum(1)
        dqn_loss = (elementwise_loss * w).mean()

        # Optimize
        # self.optimizer.zero_grad()
        # self.icm_optimizer.zero_grad()
        # (dqn_loss + icm_loss).backward()
        # torch.nn.utils.clip_grad_norm_(list(self.online.parameters()) + list(self.icm.parameters()), 10.0)
        # self.optimizer.step()
        # self.icm_optimizer.step()

        # Optimize DQN and ICM separately
        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), 10.0)
        self.icm_optimizer.step()

        self.online.reset_noise()
        self.target.reset_noise()
        self.buffer.update_priorities(indices, elementwise_loss.abs().detach().cpu().numpy())

        # Soft target update
        for target_p, online_p in zip(self.target.parameters(), self.online.parameters()):
            target_p.data.copy_(TAU * online_p.data + (1.0 - TAU) * target_p.data)

        return dqn_loss.item(), icm_loss.item(), int_r.mean().item()

    def save_model(self, path):
        torch.save({
            'online': self.online.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'icm': self.icm.state_dict(),
            'icm_optimizer': self.icm_optimizer.state_dict(),
            'frame_idx': self.frame_idx,
            'rewards': self.rewards,
            'dqn_losses': self.dqn_losses,
            'icm_losses': self.icm_losses,
            'eval_rewards': self.eval_rewards,
            'int_rewards': self.int_rewards,
            'best_eval_reward': self.best_eval_reward,
            'buffer_state': self.buffer.get_state(),
        }, path, pickle_protocol=4)

    def load_model(self, path, eval_mode=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(checkpoint['online'])
        self.target.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.icm.load_state_dict(checkpoint['icm'])
        self.icm_optimizer.load_state_dict(checkpoint['icm_optimizer'])
        self.frame_idx = checkpoint['frame_idx']
        self.rewards = checkpoint['rewards']
        self.dqn_losses = checkpoint['dqn_losses']
        self.icm_losses = checkpoint['icm_losses']
        self.eval_rewards = checkpoint.get('eval_rewards', [])
        self.int_rewards = checkpoint.get('int_rewards', [])
        self.best_eval_reward = checkpoint.get('best_eval_reward', -np.inf)
        self.buffer.set_state(checkpoint['buffer_state'])
        if eval_mode:
            self.online.eval()
            self.icm.eval()

# -----------------------------
# 7. Training Loop
# -----------------------------
def plot_figure(agent: Agent, episode: int):
    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    avg_reward = np.mean(agent.rewards[-PLOT_INTERVAL:]) if len(agent.rewards) >= PLOT_INTERVAL else np.mean(agent.rewards)
    plt.title(f"Episode {episode} | Avg Reward {avg_reward:.2f}")
    plt.plot(agent.rewards, label='Reward')
    plt.plot([i * EVAL_INTERVAL for i in range(1, len(agent.eval_rewards) + 1)], agent.eval_rewards, label='Eval Reward')
    plt.legend()
    plt.subplot(122)
    plt.title("Loss")
    plt.plot(agent.icm_losses, label='ICM Loss')
    plt.plot(agent.dqn_losses, label='DQN Loss')
    plt.legend()
    save_path = os.path.join(PLOT_DIR, f"episode_{episode}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    tqdm.write(f"Plot saved to {save_path}")

def evaluation(agent: Agent, episode: int, best_checkpoint_path='models/best.pth'):
    agent.online.eval()
    eval_env = make_env()
    state = eval_env.reset()
    e_reward = 0
    done = False
    while not done:
        e_action = agent.act(state)
        state, reward, done, _ = eval_env.step(e_action)
        e_reward += reward
    eval_env.close()
    agent.eval_rewards.append(e_reward)
    if e_reward > agent.best_eval_reward:
        agent.best_eval_reward = e_reward
    tqdm.write(f"Eval Reward: {e_reward:.1f} | Best Eval Reward: {agent.best_eval_reward:.1f}")

    if e_reward >= 4000 and e_reward == agent.best_eval_reward:
        agent.save_model(best_checkpoint_path)
        tqdm.write(f"Best model saved at episode {episode} with Eval Reward {e_reward:.1f}")

def train(num_episodes: int, checkpoint_path='models/rainbow_icm.pth', best_checkpoint_path='models/best.pth'):
    env = make_env()
    agent = Agent(env.observation_space.shape, env.action_space.n)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    start_ep = 1
    if os.path.isfile(checkpoint_path):
        agent.load_model(checkpoint_path)
        start_ep = len(agent.rewards) + 1

    agent.online.train()

    # Warm-up
    state = env.reset()
    while agent.buffer.size < BATCH_SIZE:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.buffer.store(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

    # Training
    progress_bar = tqdm(total=MAX_FRAMES, desc="Training")
    progress_bar.update(len(agent.dqn_losses))

    for ep in range(start_ep, num_episodes + 1):
        state = env.reset()
        # ep_reward = 0
        ep_env_reward = 0
        # prev_x = None
        prev_life = None
        done = False
        while not done:
            agent.frame_idx += 1
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            truncated = info.get('TimeLimit.truncated', False)
            done_flag = done and not truncated
            custom_reward = reward
            # x_pos = info['x_pos']
            # if x_pos is not None:
            #     if prev_x is None:
            #         prev_x = x_pos
            #     dx = x_pos - prev_x
            #     custom_reward += BACKWARD_PENALTY if dx < 0 else STAY_PENALTY if dx == 0 else 0
            #     prev_x = x_pos
            life = info['life']
            if prev_life is None:
                prev_life = life
            elif life < prev_life:
                custom_reward += DEATH_PENALTY
                prev_life = life
            agent.buffer.store(state, action, custom_reward, next_state, done_flag)
            dqn_loss, icm_loss, int_reward = agent.learn()
            if dqn_loss is not None:
                agent.dqn_losses.append(dqn_loss)
                agent.icm_losses.append(icm_loss)
                agent.int_rewards.append(int_reward)
            state = next_state
            # ep_reward += custom_reward
            ep_env_reward += reward
            progress_bar.update(1)
            if agent.frame_idx >= MAX_FRAMES:
                break

        agent.rewards.append(ep_env_reward)
        status = "TERMINATED" if done_flag else "TRUNCATED"

        # Logging
        if ep % PLOT_INTERVAL == 0:
            avg_reward = np.mean(agent.rewards[-PLOT_INTERVAL:]) if len(agent.rewards) >= PLOT_INTERVAL else np.mean(agent.rewards)
            tqdm.write(f"Episode {ep} | Reward {ep_env_reward:.1f} | Avg Reward {avg_reward:.1f} | Stage {env.unwrapped._stage} | Status {status}")

        # Evaluation
        if ep % EVAL_INTERVAL == 0:
            evaluation(agent, ep, best_checkpoint_path)
            agent.online.train()

        # Plot
        if ep % PLOT_INTERVAL == 0:
            plot_figure(agent, ep)

        # Save model
        if ep % SAVE_INTERVAL == 0:
            agent.save_model(checkpoint_path)
            tqdm.write(f"Model saved at episode {ep}")

        if agent.frame_idx >= MAX_FRAMES:
            break

    progress_bar.close()
    env.close()

if __name__ == '__main__':
    train(num_episodes=10000)
