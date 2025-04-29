import os
from collections import deque, namedtuple
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as U
from tqdm import tqdm
from numba import njit

from env_wrapper import make_env
from segment_tree import SumSegmentTree, MinSegmentTree, _sample_core

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

# -----------------------------
# Hyperparameters
# -----------------------------
# Env Wrappers
SKIP_FRAMES             = 4
STACK_FRAMES            = 4
MAX_EPISODE_STEPS       = 1000

# Agent
TARGET_UPDATE           = 1000
TAU                     = 0.25
LEARNING_RATE           = 1e-5
ADAM_EPS                = 0.00015

# Noisy Linear Layer
NOISY_STD_INIT          = 0.5

# Prioritized Replay Buffer
MEMORY_SIZE             = 30000
BATCH_SIZE              = 128
GAMMA                   = 0.99
N_STEP                  = 5
ALPHA                   = 0.6
BETA_START              = 0.4
BETA_FRAMES             = 2000000
MAX_FRAMES              = 2000000
PRIOR_EPS               = 1e-6
GAMMA_POW_N_STEP = GAMMA ** N_STEP

# Customized Reward
VELOCITY_REWARD         = 0.01
BACKWARD_PENALTY        = -1
TRUNCATE_PENALTY        = -100
STUCK_PENALTY           = -1
STUCK_PENALTY_STEP      = 150
STUCK_TRUNCATE_STEP     = 300

# Epsilon-Greedy
EPSILON_START           = 1.0
EPSILON_MIN             = 0.001
EPSILON_DECAY           = 0.975 # per episode

# Output
EVAL_INTERVAL           = 30
SAVE_INTERVAL           = 150
PLOT_INTERVAL           = 30
CHECK_PARAM_INTERVAL    = 150
CHECK_GRAD_INTERVAL     = 150
MODEL_DIR               = "./models"
PLOT_DIR                = "./plots"

# -----------------------------
# Noisy Linear Layer
# -----------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init=NOISY_STD_INIT):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        self.weight_mu    = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma   = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))
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
# Double Dueling Deep Recurrent Q Network
# -----------------------------
class DDDRQN(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.n_actions = n_actions

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

        self.adv_hidden_layer = NoisyLinear(512, 512)
        self.advantage_layer  = NoisyLinear(512, self.n_actions)
        self.v_hidden_layer   = NoisyLinear(512, 512)
        self.value_layer      = NoisyLinear(512, 1)

        # 使用 Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def get_features(self, x):
        return self.feature_layer(x)

    def forward(self, x):
        x = self.get_features(x)

        advantage = F.relu(self.adv_hidden_layer(x))
        advantage = self.advantage_layer(advantage)

        value = F.relu(self.v_hidden_layer(x))
        value = self.value_layer(value)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        self.adv_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.v_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

# -----------------------------
# Prioritized Replay Buffer
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
def _get_beta_by_frame(frame_idx):
    return min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

class PrioritizedReplayBuffer:
    def __init__(self, obs_shape: Tuple):
        self.obs_shape    = obs_shape
        self.obs_buf      = np.zeros((MEMORY_SIZE,) + obs_shape, dtype=np.float32)
        self.next_obs_buf = np.zeros((MEMORY_SIZE,) + obs_shape, dtype=np.float32)
        self.acts_buf     = np.zeros((MEMORY_SIZE,), dtype=np.int32)
        self.rews_buf     = np.zeros((MEMORY_SIZE,), dtype=np.float32)
        self.done_buf     = np.zeros((MEMORY_SIZE,), dtype=np.float32)
        self.size = 0
        self.ptr  = 0
        self.n_step_buffer = deque(maxlen=N_STEP)
        self.max_priority = 1.0
        tree_capacity = 1
        while tree_capacity < MEMORY_SIZE:
            tree_capacity <<= 1 # *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, state, action, reward, next_state, done):
        self.n_step_buffer.append(Transition(state, action, reward, next_state, done))
        if len(self.n_step_buffer) < N_STEP:
            return
        reward_n, next_state_n, done_n = self._get_n_step()
        self.obs_buf[self.ptr]       = self.n_step_buffer[0].state
        self.acts_buf[self.ptr]      = self.n_step_buffer[0].action
        self.rews_buf[self.ptr]      = reward_n
        self.next_obs_buf[self.ptr]  = next_state_n
        self.done_buf[self.ptr]      = done_n
        self.sum_tree[self.ptr]      = self.max_priority ** ALPHA
        self.min_tree[self.ptr]      = self.max_priority ** ALPHA
        self.ptr                     = (self.ptr + 1) % MEMORY_SIZE
        self.size                    = min(self.size + 1, MEMORY_SIZE)

    def _get_n_step(self):
        # 提取 n_step_buffer 的資料為 NumPy 陣列
        rewards         = np.array([trans.reward for trans in self.n_step_buffer], dtype=np.float32)
        dones           = np.array([trans.done for trans in self.n_step_buffer], dtype=np.float32)
        last_next_state = self.n_step_buffer[-1].next_state
        last_done       = float(self.n_step_buffer[-1].done)
        # 用 numba 加速計算
        reward, next_state, done = _compute_n_step_return(rewards, dones, last_next_state, last_done, GAMMA)
        return reward, next_state, done

    def sample(self, frame_idx: int):
        if self.size < BATCH_SIZE:
            return None
        indices, weights = _sample_core(
            sum_tree   = self.sum_tree.tree,
            min_tree   = self.min_tree.tree,
            size       = self.size,
            batch_size = BATCH_SIZE,
            beta       = _get_beta_by_frame(frame_idx),
            capacity   = self.sum_tree.capacity
        )
        batch = Transition(
            state      = self.obs_buf[indices],
            action     = self.acts_buf[indices],
            reward     = self.rews_buf[indices],
            next_state = self.next_obs_buf[indices],
            done       = self.done_buf[indices]
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
        self.n_step_buffer = deque(state['n_step_buffer'], maxlen=N_STEP)
        self.sum_tree.tree = state['sum_tree']
        self.min_tree.tree = state['min_tree']
        self.max_priority  = state['max_priority']

# -----------------------------
# Agent
# -----------------------------
class Agent:
    def __init__(self, obs_shape: Tuple, n_actions: int, device=None):
        self.device    = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions = n_actions

        self.online    = DDDRQN(obs_shape[0], n_actions).to(self.device)
        self.target    = DDDRQN(obs_shape[0], n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

        self.buffer = PrioritizedReplayBuffer(obs_shape)

        # 初始化損失函數
        self.dqn_criterion     = nn.SmoothL1Loss(reduction='none')  # 用於 DQN Loss，逐元素計算

        self.epsilon           = EPSILON_START
        self.frame_idx         = 0
        self.rewards           = []
        self.custom_rewards    = []
        self.dqn_losses        = []
        self.eval_rewards      = []
        self.best_eval_reward  = -np.inf

    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_value = self.online(state_tensor)
        return int(q_value.argmax(dim=1).item())

    def check_parameters(self):
        """檢查 online、target 模塊的參數值大小"""
        models = {
            "online": self.online,
            "target": self.target,
        }

        max_len = len("adv_hidden_layer.weight_sigma")

        for model_name, model in models.items():
            param_stats = []
            for name, param in model.named_parameters():
                # 計算參數的統計值
                param_norm = param.norm().item()  # L2 範數
                param_mean = param.mean().item()  # 平均值
                param_std  = param.std().item()   # 標準差
                param_max  = param.max().item()   # 最大值
                param_min  = param.min().item()   # 最小值
                param_stats.append(
                    f"{name}:{' ' * (max_len - len(name))}"
                    f"norm={' ' if param_norm >= 0 else ''}{param_norm:02.5f},\t"
                    f"mean={' ' if param_mean >= 0 else ''}{param_mean:.5f},\t"
                    f"std={' ' if param_std >= 0 else ''}{param_std:.5f},\t"
                    f"max={' ' if param_max >= 0 else ''}{param_max:.5f},\t"
                    f"min={' ' if param_min >= 0 else ''}{param_min:.5f}"
                )

            # 打印統計信息
            print(f"\n[{model_name}] Parameter Statistics at Frame {self.frame_idx}:")
            for stat in param_stats:
                print(stat)

    def check_gradients(self):
        """檢查梯度大小"""
        models = {
            "online": self.online,
        }

        max_len = len("adv_hidden_layer.weight_sigma")

        for model_name, model in models.items():
            grad_stats = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_mean = param.grad.mean().item()
                    grad_std  = param.grad.std().item()
                    grad_max  = param.grad.max().item()
                    grad_min  = param.grad.min().item()
                    grad_stats.append(
                        f"{name}:{' ' * (max_len - len(name))}"
                        f"grad_norm={' ' if grad_norm >= 0 else ''}{grad_norm:.5f},\t"
                        f"grad_mean={' ' if grad_mean >= 0 else ''}{grad_mean:.5f},\t"
                        f"grad_std={' ' if grad_std >= 0 else ''}{grad_std:.5f},\t"
                        f"grad_max={' ' if grad_max >= 0 else ''}{grad_max:.5f},\t"
                        f"grad_min={' ' if grad_min >= 0 else ''}{grad_min:.5f}"
                    )
            print(f"\n[{model_name}] Gradient Statistics at Frame {self.frame_idx}:")
            for stat in grad_stats:
                print(stat)

    def learn(self):
        batch, weights, indices = self.buffer.sample(self.frame_idx)
        states      = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        actions     = torch.tensor(batch.action, dtype=torch.int64, device=self.device)
        rewards     = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        dones       = torch.tensor(batch.done, dtype=torch.float32, device=self.device)
        weights     = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # DQN targets
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(1)
            q_next       = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_target     = rewards + GAMMA_POW_N_STEP * q_next * (1 - dones)
        q_predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_value    = q_predicted - q_target.detach()
        dqn_loss    = (self.dqn_criterion(q_predicted, q_target.detach()) * weights).mean()

        # update DQN
        self.optimizer.zero_grad()
        dqn_loss.backward()
        U.clip_grad_norm_(self.online.parameters(), 5.0)
        # U.clip_grad_value_(self.online.parameters(), 1.0)
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_value.abs().detach().cpu().numpy())

        if self.frame_idx % TARGET_UPDATE == 0:
            if TAU == 1.0:
                # Hard target update
                self.target.load_state_dict(self.online.state_dict())
            else:
                # Soft target update
                for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
                    target_param.data.copy_(TAU * online_param.data + (1.0 - TAU) * target_param.data)

        return dqn_loss.item()

    def save_model(self, path):
        torch.save({
            'online'           : self.online.state_dict(),
            'target'           : self.target.state_dict(),
            'optimizer'        : self.optimizer.state_dict(),
            'epsilon'          : self.epsilon,
            'frame_idx'        : self.frame_idx,
            'rewards'          : self.rewards,
            'custom_rewards'   : self.custom_rewards,
            'dqn_losses'       : self.dqn_losses,
            'eval_rewards'     : self.eval_rewards,
            'best_eval_reward' : self.best_eval_reward,
            'buffer_state'     : self.buffer.get_state(),
        }, path, pickle_protocol=4)

    def load_model(self, path, eval_mode=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(checkpoint['online'])
        if eval_mode:
            self.online.eval()
            return
        self.target.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon           = checkpoint.get('epsilon', EPSILON_START)
        self.frame_idx         = checkpoint.get('frame_idx', 0)
        self.rewards           = checkpoint.get('rewards', [])
        self.custom_rewards    = checkpoint.get('custom_rewards', [])
        self.dqn_losses        = checkpoint.get('dqn_losses', [])
        self.eval_rewards      = checkpoint.get('eval_rewards', [])
        self.best_eval_reward  = checkpoint.get('best_eval_reward', -np.inf)
        self.buffer.set_state(checkpoint['buffer_state'])

# -----------------------------
# Training Loop
# -----------------------------
def plot_figure(agent: Agent, episode: int):
    plt.figure(figsize=(15, 15))

    plt.subplot(311)
    avg_reward = np.mean(agent.rewards[-PLOT_INTERVAL:]) if len(agent.rewards) >= PLOT_INTERVAL else np.mean(agent.rewards)
    plt.title(f"Life {episode} | Avg Reward {avg_reward:.1f}")
    plt.plot(1 + np.arange(len(agent.custom_rewards)), agent.custom_rewards, label='Custom Reward')
    plt.plot(1 + np.arange(len(agent.rewards)), agent.rewards, label='Reward')
    plt.xlim(left=1, right=len(agent.rewards))
    plt.ylim(bottom=max(-1000.0, min(agent.rewards), min(agent.custom_rewards)))
    plt.legend()

    arr = np.array(agent.rewards)
    # 補 0 讓長度是3的倍數
    if len(arr) % 3 != 0:
        arr = np.pad(arr, (0, 3 - len(arr) % 3))
    episode_rewards = arr.reshape(-1, 3).sum(axis=1)

    plt.subplot(312)
    avg_reward = np.mean(episode_rewards[-PLOT_INTERVAL//3:]) if len(episode_rewards) >= PLOT_INTERVAL // 3 else np.mean(episode_rewards)
    plt.title(f"Episode {episode // 3} | Avg Reward {avg_reward:.1f}")
    plt.plot(1 + np.arange(len(episode_rewards)), episode_rewards, label='Reward')
    plt.plot((1 + np.arange(len(agent.eval_rewards))) * EVAL_INTERVAL // 3, agent.eval_rewards, label='Eval Reward')
    plt.xlim(left=1, right=len(episode_rewards))
    plt.ylim(bottom=max(-1000.0, min(episode_rewards), min(agent.eval_rewards)))
    plt.legend()

    plt.subplot(313)
    plt.title("DQN Loss")
    plt.plot(agent.dqn_losses, label='DQN Loss')
    plt.xlim(left=0.0, right=len(agent.dqn_losses))
    plt.ylim(bottom=0.0,top=np.max(agent.dqn_losses[-int(0.9 * len(agent.dqn_losses)):] if len(agent.rewards) >= PLOT_INTERVAL else agent.dqn_losses))
    # plt.legend()

    save_path = os.path.join(PLOT_DIR, f"episode_{episode}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    tqdm.write(f"Plot saved to {save_path}")

def evaluation(agent: Agent, episode: int, best_checkpoint_path='models/best.pth'):
    with torch.no_grad():
        agent.online.eval()
        eval_env = make_env(SKIP_FRAMES, STACK_FRAMES, 3 * MAX_EPISODE_STEPS, False)
        state = eval_env.reset()
        eval_reward = 0
        done = False
        while not done:
            e_action = agent.act(state)
            state, reward, done, _ = eval_env.step(e_action)
            eval_reward += reward
        eval_env.close()
        agent.eval_rewards.append(eval_reward)
        if eval_reward > agent.best_eval_reward:
            agent.best_eval_reward = eval_reward
        tqdm.write(f"Eval Reward: {eval_reward:.0f} | Best Eval Reward: {agent.best_eval_reward:.0f}")

        if eval_reward >= 3000 and eval_reward == agent.best_eval_reward:
            agent.save_model(best_checkpoint_path)
            tqdm.write(f"Best model saved at episode {episode} with Eval Reward {eval_reward:.0f}")

def learn_human_play(agent: Agent):
    import pickle

    ids = [40, 41, 42, 43, 44]

    for id in ids:
        path = f"./human_play/play_{id}.pkl"
        with open(path, 'rb') as f:
            play = pickle.load(f)
        episode_reward = play['total_reward']
        trajectory = play['trajectory']

        for state, action, reward, next_state, done in trajectory:
            agent.buffer.store(state, action, reward, next_state, done)

            if agent.buffer.size < BATCH_SIZE:
                continue

            dqn_loss = agent.learn()
            agent.dqn_losses.append(dqn_loss)

        tqdm.write(f"Learn from {path}, total reward: {episode_reward}, frames: {len(trajectory)}")

def train(num_episodes: int, checkpoint_path='models/rainbow_icm.pth', best_checkpoint_path='models/best.pth'):
    env = make_env(SKIP_FRAMES, STACK_FRAMES, MAX_EPISODE_STEPS, True)
    agent = Agent(env.observation_space.shape, env.action_space.n)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    start_episode = 1
    if os.path.isfile(checkpoint_path):
        agent.load_model(checkpoint_path)
        start_episode = len(agent.rewards) + 1

    agent.online.train()

    # learn_human_play(agent)

    # Warm-up
    state = env.reset()
    while agent.buffer.size < BATCH_SIZE:
        action = np.random.randint(env.action_space.n)
        next_state, reward, done, _ = env.step(action)
        agent.buffer.store(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

    # Training
    progress_bar = tqdm(total=MAX_FRAMES, desc="Training")
    progress_bar.update(len(agent.dqn_losses))

    count_truncated = 0

    for episode in range(start_episode, num_episodes + 1):
        state                 = env.reset()
        episode_reward        = 0
        episode_custom_reward = 0
        steps                 = 0
        stuck_steps           = 0
        prev_x                = None
        done                  = False
        truncated             = False

        agent.online.reset_noise()
        agent.target.reset_noise()

        while not done:
            agent.frame_idx += 1
            steps += 1

            if np.random.rand() < agent.epsilon:
                action = np.random.randint(agent.n_actions)
            else:
                action = agent.act(state)

            next_state, reward, done, info = env.step(action)
            truncated = info.get('TimeLimit.truncated', False)

            # Reward Shaping
            custom_reward = reward

            x_pos = info['x_pos']
            if prev_x is None:
                prev_x = x_pos
            dx = x_pos - prev_x
            prev_x = x_pos

            if dx <= 1:
                stuck_steps += 1
            else:
                stuck_steps = 0

            if stuck_steps >= STUCK_PENALTY_STEP:
                custom_reward += STUCK_PENALTY
            if stuck_steps >= STUCK_TRUNCATE_STEP:
                truncated = True
                done = True

            if dx < 0:
                custom_reward += BACKWARD_PENALTY
            # else:
            #     custom_reward += VELOCITY_REWARD * dx
            v = episode_reward / steps
            if v >= 3.0:
                custom_reward += VELOCITY_REWARD * v

            if truncated:
                custom_reward += TRUNCATE_PENALTY

            # custom_reward = np.sign(custom_reward) * (np.sqrt(np.abs(custom_reward) + 1) - 1) + custom_reward / 12.0

            agent.buffer.store(state, action, custom_reward, next_state, done or truncated)

            dqn_loss = agent.learn()
            agent.dqn_losses.append(dqn_loss)

            state = next_state
            episode_custom_reward += custom_reward
            episode_reward += reward
            progress_bar.update(1)
            if agent.frame_idx >= MAX_FRAMES:
                break

        agent.rewards.append(episode_reward)
        agent.custom_rewards.append(episode_custom_reward)
        if truncated:
            count_truncated += 1

        # Check parameters
        if CHECK_PARAM_INTERVAL > 0 and episode % CHECK_PARAM_INTERVAL == 0:
            agent.check_parameters()

        # Check gradients
        if CHECK_GRAD_INTERVAL > 0 and episode % CHECK_GRAD_INTERVAL == 0:
            agent.check_gradients()

        # Logging
        tqdm.write(f"Episode {episode}\t| Steps {steps}\t| Reward {episode_reward:.0f}\t| Custom Reward {episode_custom_reward:.1f}\t| Stage {env.unwrapped._stage} | Truncated {truncated}\t| Epsilon {agent.epsilon:.4f}")

        # Epsilon Decay
        agent.epsilon = max(agent.epsilon * EPSILON_DECAY, EPSILON_MIN)

        # Logging
        if episode % PLOT_INTERVAL == 0:
            avg_reward = np.mean(agent.rewards[-PLOT_INTERVAL:] if len(agent.rewards) >= PLOT_INTERVAL else agent.rewards)
            avg_custom_reward = np.mean(agent.custom_rewards[-PLOT_INTERVAL:] if len(agent.custom_rewards) >= PLOT_INTERVAL else agent.custom_rewards)
            tqdm.write(f"Avg Reward {avg_reward:.1f} | Avg Custom Reward {avg_custom_reward:.1f} | Truncated {count_truncated}")
            count_truncated = 0

        # Evaluation
        if episode % EVAL_INTERVAL == 0:
            evaluation(agent, episode, best_checkpoint_path)
            agent.online.train()

        # Plot
        if episode % PLOT_INTERVAL == 0:
            plot_figure(agent, episode)

        # Save model
        if episode % SAVE_INTERVAL == 0:
            agent.save_model(checkpoint_path)
            tqdm.write(f"Model saved at episode {episode}")

        if agent.frame_idx >= MAX_FRAMES:
            break

    progress_bar.close()
    env.close()

if __name__ == '__main__':
    train(num_episodes=10000)
