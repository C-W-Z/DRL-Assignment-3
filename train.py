import os
from collections import deque, namedtuple
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
MAX_EPISODE_STEPS       = 3000

# Agent
TARGET_UPDATE           = 1000
TAU                     = 0.9
LEARNING_RATE           = 0.0001
ADAM_EPS                = 0.00015

# Noisy Linear Layer
NOISY_STD_INIT          = 2.5

# Prioritized Replay Buffer
MEMORY_SIZE             = 50000
BATCH_SIZE              = 32
GAMMA                   = 0.95
N_STEP                  = 5
ALPHA                   = 0.6
BETA_START              = 0.4
BETA_FRAMES             = 2000000
MAX_FRAMES              = 2000000
PRIOR_EPS               = 1e-6
GAMMA_POW_N_STEP = GAMMA ** N_STEP

# Customized Reward
BACKWARD_PENALTY        = 0
STAY_PENALTY            = 0
DEATH_PENALTY           = -100

# Intrinsic Curiosity Module
ICM_BETA                = 0.2
ICM_ETA                 = 0.05
ICM_LR                  = 1e-4
ICM_EMBED_DIM           = 256

# Output
EVAL_INTERVAL           = 10
SAVE_INTERVAL           = 100
PLOT_INTERVAL           = 10
CHECK_PARAM_INTERVAL    = 10
CHECK_GRAD_INTERVAL     = 10
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
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
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
# Intrinsic Curiosity Module
# -----------------------------
class ICM(nn.Module):
    def __init__(self, feature_dimension: int, n_actions: int, embed_dimension: int=ICM_EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dimension, embed_dimension),
            nn.ReLU(),
            nn.Linear(embed_dimension, embed_dimension),
            nn.ReLU()
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(embed_dimension * 2, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(embed_dimension + n_actions, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dimension)
        )

    def forward(self, features, next_features, action):
        phi           = self.encoder(features)
        phi_next      = self.encoder(next_features)
        inverse_input = torch.cat([phi, phi_next], dim=1)
        logits        = self.inverse_model(inverse_input)
        action_onehot = F.one_hot(action, logits.size(-1)).float()
        forward_input = torch.cat([phi, action_onehot], dim=1)
        pred_phi_next = self.forward_model(forward_input)
        return logits, pred_phi_next, phi_next

# -----------------------------
# Dueling Network
# -----------------------------
class DuelingNetwork(nn.Module):
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
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            self.feature_dimension  = self.feature_layer(dummy).shape[1]

        self.adv_hidden_layer = NoisyLinear(self.feature_dimension, 512)
        self.advantage_layer  = NoisyLinear(512, self.n_actions)
        self.v_hidden_layer   = NoisyLinear(self.feature_dimension, 512)
        self.value_layer      = NoisyLinear(512, 1)

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
        self.tree_ptr = 0
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
        self.sum_tree[self.tree_ptr] = self.max_priority ** ALPHA
        self.min_tree[self.tree_ptr] = self.max_priority ** ALPHA
        self.tree_ptr = (self.tree_ptr + 1) % MEMORY_SIZE
        self.ptr = (self.ptr + 1) % MEMORY_SIZE
        self.size = min(self.size + 1, MEMORY_SIZE)

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
            'tree_ptr'     : self.tree_ptr,
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
        self.tree_ptr      = state['tree_ptr']
        self.max_priority  = state['max_priority']

# -----------------------------
# Agent
# -----------------------------
class Agent:
    def __init__(self, obs_shape: Tuple, n_actions: int, device=None):
        self.device    = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions = n_actions

        self.online    = DuelingNetwork(obs_shape[0], n_actions).to(self.device)
        self.target    = DuelingNetwork(obs_shape[0], n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

        self.icm           = ICM(self.online.feature_dimension, n_actions).to(self.device)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=ICM_LR)

        self.buffer = PrioritizedReplayBuffer(obs_shape)

        # 初始化損失函數
        self.dqn_criterion     = nn.SmoothL1Loss(reduction='none')  # 用於 DQN Loss，逐元素計算
        self.inverse_criterion = nn.CrossEntropyLoss(reduction='mean')  # 用於 ICM 的逆向損失
        self.forward_criterion = nn.MSELoss(reduction='mean')  # 用於 ICM 的前向損失

        self.frame_idx         = 0
        self.rewards           = []
        self.dqn_losses        = []
        self.icm_losses        = []
        self.eval_rewards      = []
        self.intrinsic_rewards = []
        self.best_eval_reward  = -np.inf

    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_value = self.online(state_tensor)
        return int(q_value.argmax(1).item())

    def check_parameters(self):
        """檢查 online、target 和 icm 模塊的參數值大小"""
        models = {
            "online": self.online,
            "target": self.target,
            "icm": self.icm
        }

        max_len = len("advantage_layer.weight_sigma")

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
                    f"{name}:{' ' * (max_len - len(name))}norm={param_norm:.4f},\tmean={param_mean:.4f},\t"
                    f"std={param_std:.4f},\tmax={param_max:.4f},\tmin={param_min:.4f}"
                )

            # 打印統計信息
            print(f"[{model_name}] Parameter Statistics at Frame {self.frame_idx}:")
            for stat in param_stats:
                print(stat)

    def check_gradients(self):
        """檢查梯度大小"""
        models = {
            "online": self.online,
            "icm": self.icm
        }

        max_len = len("advantage_layer.weight_sigma")

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
                        f"{name}:{' ' * (max_len - len(name))}grad_norm={grad_norm:.4f},\tgrad_mean={grad_mean:.4f},\t"
                        f"grad_std={grad_std:.4f},\tgrad_max={grad_max:.4f},\tgrad_min={grad_min:.4f}"
                    )
            print(f"\n[{model_name}] Gradient Statistics at Frame {self.frame_idx}:")
            for stat in grad_stats:
                print(stat)

    def learn(self):
        batch, weights, indices = self.buffer.sample(self.frame_idx)
        states           = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        actions          = torch.tensor(batch.action, dtype=torch.int64, device=self.device)
        external_rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_states      = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        dones            = torch.tensor(batch.done, dtype=torch.float32, device=self.device)
        weights          = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # features
        features      = self.online.get_features(states)
        next_features = self.online.get_features(next_states)

        # ICM forward
        logits, predicted_phi_next, true_phi_next = self.icm(features, next_features, actions)
        inverse_loss = self.inverse_criterion(logits, actions)
        forward_loss = self.forward_criterion(predicted_phi_next, true_phi_next.detach())
        icm_loss     = (1 - ICM_BETA) * inverse_loss + ICM_BETA * forward_loss
        with torch.no_grad():
            intrinsic_rewards = ICM_ETA * 0.5 * (predicted_phi_next - true_phi_next).pow(2).sum(dim=1)
            intrinsic_rewards = intrinsic_rewards / (intrinsic_rewards.mean() + 1e-6) * external_rewards.mean()

        # DQN targets
        q_predicted   = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions  = self.online(next_states).argmax(1)
        q_next        = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        total_rewards = external_rewards + intrinsic_rewards
        q_target      = total_rewards + GAMMA_POW_N_STEP * q_next * (1 - dones)
        td_value      = q_predicted - q_target.detach()
        dqn_loss      = (self.dqn_criterion(q_predicted, q_target.detach()) * weights).mean()
        # dqn_loss, elementwise_loss = self.compute_dqn_loss(states, actions, total_rewards, next_states, dones, weights)

        # update DQN & ICM
        self.optimizer.zero_grad()
        self.icm_optimizer.zero_grad()
        dqn_loss.backward()
        icm_loss.backward()
        self.optimizer.step()
        self.icm_optimizer.step()

        self.online.reset_noise()
        self.target.reset_noise()
        self.buffer.update_priorities(indices, td_value.abs().detach().cpu().numpy())

        if self.frame_idx % TARGET_UPDATE == 0:
            if TAU == 1.0:
                # Hard target update
                self.target.load_state_dict(self.online.state_dict())
            else:
                # Soft target update
                for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
                    target_param.data.copy_(TAU * online_param.data + (1.0 - TAU) * target_param.data)

        return dqn_loss.item(), icm_loss.item(), intrinsic_rewards.mean().item() # 返回損失值和內在獎勵

    def save_model(self, path):
        torch.save({
            'online'           : self.online.state_dict(),
            'target'           : self.target.state_dict(),
            'optimizer'        : self.optimizer.state_dict(),
            'icm'              : self.icm.state_dict(),
            'icm_optimizer'    : self.icm_optimizer.state_dict(),
            'frame_idx'        : self.frame_idx,
            'rewards'          : self.rewards,
            'dqn_losses'       : self.dqn_losses,
            'icm_losses'       : self.icm_losses,
            'eval_rewards'     : self.eval_rewards,
            'intrinsic_rewards': self.intrinsic_rewards,
            'best_eval_reward' : self.best_eval_reward,
            'buffer_state'     : self.buffer.get_state(),
        }, path, pickle_protocol=4)

    def load_model(self, path, eval_mode=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(checkpoint['online'])
        self.target.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.icm.load_state_dict(checkpoint['icm'])
        self.icm_optimizer.load_state_dict(checkpoint['icm_optimizer'])
        self.frame_idx         = checkpoint['frame_idx']
        self.rewards           = checkpoint['rewards']
        self.dqn_losses        = checkpoint.get('dqn_losses', [])
        self.icm_losses        = checkpoint.get('icm_losses', [])
        self.eval_rewards      = checkpoint.get('eval_rewards', [])
        self.intrinsic_rewards = checkpoint.get('intrinsic_rewards', [])
        self.best_eval_reward  = checkpoint.get('best_eval_reward', -np.inf)
        self.buffer.set_state(checkpoint['buffer_state'])
        if eval_mode:
            self.online.eval()
            self.icm.eval()

# -----------------------------
# Training Loop
# -----------------------------
def plot_figure(agent: Agent, episode: int):
    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    avg_reward = np.mean(agent.rewards[-PLOT_INTERVAL:]) if len(agent.rewards) >= PLOT_INTERVAL else np.mean(agent.rewards)
    plt.title(f"Episode {episode} | Avg Reward {avg_reward:.2f}")
    plt.plot(1 + np.arange(len(agent.rewards)), agent.rewards, label='Reward')
    plt.plot((1 + np.arange(len(agent.eval_rewards))) * EVAL_INTERVAL, agent.eval_rewards, label='Eval Reward')
    plt.xlim(left=1, right=len(agent.rewards))
    plt.legend()
    plt.subplot(122)
    plt.title("Loss")
    plt.plot(agent.icm_losses, label='ICM Loss')
    plt.plot(agent.dqn_losses, label='DQN Loss')
    plt.xlim(left=0.0, right=len(agent.dqn_losses))
    plt.ylim(bottom=0.0,top=max(agent.dqn_losses[-10 * TARGET_UPDATE:] if len(agent.dqn_losses) >= 10 * TARGET_UPDATE else agent.dqn_losses))
    plt.legend()
    save_path = os.path.join(PLOT_DIR, f"episode_{episode}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    tqdm.write(f"Plot saved to {save_path}")

def evaluation(agent: Agent, episode: int, best_checkpoint_path='models/best.pth'):
    agent.online.eval()
    eval_env = make_env(SKIP_FRAMES, STACK_FRAMES, MAX_EPISODE_STEPS)
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

    if eval_reward >= 4000 and eval_reward == agent.best_eval_reward:
        agent.save_model(best_checkpoint_path)
        tqdm.write(f"Best model saved at episode {episode} with Eval Reward {eval_reward:.0f}")

def train(num_episodes: int, checkpoint_path='models/rainbow_icm.pth', best_checkpoint_path='models/best.pth'):
    env = make_env(SKIP_FRAMES, STACK_FRAMES, MAX_EPISODE_STEPS)
    agent = Agent(env.observation_space.shape, env.action_space.n)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    start_episode = 1
    if os.path.isfile(checkpoint_path):
        agent.load_model(checkpoint_path)
        start_episode = len(agent.rewards) + 1

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

    for episode in range(start_episode, num_episodes + 1):
        state = env.reset()
        # ep_reward = 0
        episode_reward = 0
        # prev_x = None
        prev_life = None
        done = False
        count_truncated = 0

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
                agent.intrinsic_rewards.append(int_reward)

            state = next_state
            # ep_reward += custom_reward
            episode_reward += reward
            progress_bar.update(1)
            if agent.frame_idx >= MAX_FRAMES:
                break

        agent.rewards.append(episode_reward)
        if not done_flag:
            count_truncated += 1

        # Check parameters
        if CHECK_PARAM_INTERVAL > 0 and episode % CHECK_PARAM_INTERVAL == 0:
            agent.check_parameters()

        # Check gradients
        if CHECK_GRAD_INTERVAL > 0 and episode % CHECK_GRAD_INTERVAL == 0:
            agent.check_gradients()

        # Logging
        if episode % PLOT_INTERVAL == 0:
            avg_reward = np.mean(agent.rewards[-PLOT_INTERVAL:]) if len(agent.rewards) >= PLOT_INTERVAL else np.mean(agent.rewards)
            tqdm.write(f"Episode {episode} | Reward {episode_reward:.0f} | Avg Reward {avg_reward:.1f} | Stage {env.unwrapped._stage} | Truncated {count_truncated}")
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
