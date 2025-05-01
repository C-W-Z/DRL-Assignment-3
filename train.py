import os
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from torch.optim import Adam
from collections import deque
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from tqdm import tqdm

from env_wrapper import make_env

# -----------------------------
# Hyperparameters
# -----------------------------
RENDER                  = False

# Env Wrappers
SKIP_FRAMES             = 4
STACK_FRAMES            = 4

# DQN
TARGET_UPDATE_FRAMES    = 5
TARGET_UPDATE_TAU       = 1e-3
DQN_LEARNING_RATE       = 1e-4
DQN_ADAM_EPS            = 1.5e-4
DQN_WEIGHT_DECAY        = 1e-6

# Intrinsic Curiosity Module
ICM_BETA                = 0.2
ICM_ETA                 = 1.0
ICM_EMBED_DIM           = 256
ICM_LEARNING_RATE       = 5e-4

# Epsilon Boltzmann Exploration
EPSILON                 = 0.1
EXPLORE_TAU             = 1.0

# Prioritized Replay Buffer
MEMORY_SIZE             = 50_000
BATCH_SIZE              = 64
GAMMA                   = 0.95
N_STEP                  = 5
ALPHA                   = 0.6
BETA_START              = 0.4
BETA_FRAMES             = 2_000_000
PRIOR_EPS               = 1e-6
GAMMA_POW_N_STEP = GAMMA ** N_STEP

# Output
EVAL_INTERVAL           = 30
SAVE_INTERVAL           = 300
PLOT_INTERVAL           = 30
CHECK_PARAM_INTERVAL    = 150
CHECK_GRAD_INTERVAL     = 150
MODEL_DIR               = "./models"
PLOT_DIR                = "./plots"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------- N-Step Replay Buffer -------

class NStepReplayBuffer:
    def __init__(self):
        self.n_step_queue = deque()

        self.buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=MEMORY_SIZE, device=DEVICE),
            batch_size=BATCH_SIZE
        )

    @property
    def size(self):
        return len(self.buffer)

    def store(self, obs, action, reward, next_obs, done):
        self.n_step_queue.append((obs, action, reward, next_obs, done))

        if len(self.n_step_queue) < N_STEP:
            return

        if len(self.n_step_queue) < N_STEP:
            return

        # 建立 n-step transition
        R, s0, a0 = 0, *self.n_step_queue[0][:2]

        g = GAMMA
        for _, _, r, _, d in self.n_step_queue:
            R += g * r
            g *= GAMMA
            if d:
                break

        # 下一狀態為最後一筆的 next_state
        _, _, _, s_n, done_n = self.n_step_queue[-1]

        data = TensorDict({
            'obs':       torch.tensor(s0, dtype=torch.float32),
            'acts':      torch.tensor([a0], dtype=torch.int64),
            'rews':      torch.tensor([R], dtype=torch.float32),
            'next_obs':  torch.tensor(s_n, dtype=torch.float32),
            'done':      torch.tensor([done_n], dtype=torch.bool),
        }, batch_size=[]).to(DEVICE)

        self.buffer.add(data)

        # 移除最前面一筆，讓 buffer 滾動
        self.n_step_queue.popleft()

        if done:
            # 清空整個 queue
            self.finish_episode()

    def finish_episode(self):
        # 用來在 episode 結束後強制 flush 剩下步數
        while self.n_step_queue:
            self.store(*self.n_step_queue.popleft())

    def sample_batch(self):
        batch = self.buffer.sample(batch_size=BATCH_SIZE)

        return {
            'obs':       batch['obs'],
            'acts':      batch['acts'].squeeze(1),
            'rews':      batch['rews'].squeeze(1),
            'next_obs':  batch['next_obs'],
            'done':      batch['done'].float().squeeze(1),
        }

    def get_state(self):
        return {
            'storage': self.buffer._storage._storage,
        }

    def set_state(self, state):
        self.buffer._storage._storage = state['storage']

# ------- Intrinsic Curiosity Module -------

class ICM(nn.Module):
    def __init__(self, feature_dimension: int, n_actions: int, embed_dimension: int=ICM_EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dimension, embed_dimension),
            nn.ReLU(),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(embed_dimension * 2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, n_actions)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(embed_dimension + n_actions, 512),
            nn.LeakyReLU(),
            nn.Linear(512, embed_dimension)
        )

    def forward(self, features, next_features, actions):
        phi           = self.encoder(features)
        phi_next      = self.encoder(next_features)

        inv_input     = torch.cat([phi, phi_next], dim=1)
        pred_action   = self.inverse_model(inv_input)

        action_onehot = F.one_hot(actions, num_classes=pred_action.size(-1)).float()
        forward_input = torch.cat([phi, action_onehot], dim=1)
        pred_phi_next = self.forward_model(forward_input)

        return pred_action, pred_phi_next, phi_next

# ------- Double Dueling Deep Recurrent Q Network -------

class D3QN(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.n_actions = n_actions

        self.feature_layer = nn.Sequential(
            # Input: (BATCH_SIZE, in_channels=4, 84, 84)
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            self.feature_dimension  = self.feature_layer(dummy).shape[1]

        self.value_layer = nn.Sequential(
            nn.Linear(self.feature_dimension, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(self.feature_dimension, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        # 使用 Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x         = self.feature_layer(x)
        value     = self.value_layer(x)
        advantage = self.advantage_layer(x)
        q         = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

# ------- Agent -------

def build_dqn_optimizer(model: nn.Module):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 排除 bias 與 norm 層參數
        if name.endswith(".bias") or "norm" in name or "bn" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = Adam(
        [
            {"params": decay_params, "weight_decay": DQN_WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=DQN_LEARNING_RATE,
        eps=DQN_ADAM_EPS
    )
    return optimizer

class Agent:
    def __init__(self, obs_shape: Tuple, n_actions: int):
        self.n_actions      = n_actions

        self.online         = D3QN(obs_shape[0], n_actions).to(DEVICE)
        self.target         = D3QN(obs_shape[0], n_actions).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.icm            = ICM(self.online.feature_dimension, n_actions).to(DEVICE)

        # self.optimizer      = Adam(self.online.parameters(), lr=DQN_LEARNING_RATE, eps=DQN_ADAM_EPS, weight_decay=DQN_WEIGHT_DECAY)
        self.optimizer      = build_dqn_optimizer(self.online)
        self.icm_optimizer  = Adam(self.icm.parameters(), lr=ICM_LEARNING_RATE, weight_decay=DQN_WEIGHT_DECAY)

        # self.buffer         = PrioritizedReplayBuffer(obs_shape, MEMORY_SIZE, BATCH_SIZE, ALPHA, N_STEP, GAMMA, PRIOR_EPS)
        self.buffer         = NStepReplayBuffer()

        self.dqn_criterion      = nn.MSELoss()
        self.forward_criterion  = nn.MSELoss()
        self.inverse_criterion  = nn.CrossEntropyLoss()

        self.frame_idx         = 0
        self.rewards           = []
        self.dqn_losses        = []
        self.forward_losses    = []
        self.inverse_losses    = []
        self.intrinsic_rewards = []
        self.eval_rewards      = []
        self.best_eval_reward  = -np.inf

    def act(self, state, deterministic=False, tau=EXPLORE_TAU):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.online(state_tensor)  # Shape: (1, n_actions)
            if not self.online.training or deterministic:
                # Greedy selection
                return int(q_values.argmax(dim=1).item())
            else:
                # Boltzmann exploration
                probabilities = F.softmax(q_values / tau, dim=1)
                action = torch.multinomial(probabilities, num_samples=1).item()
        return action

    def check_parameters(self):
        """檢查 online、target 模塊的參數值大小"""
        models = {
            "online": self.online,
            "target": self.target,
            "icm": self.icm
        }

        max_len = len("advantage_layer.0.weight:")

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
            "icm": self.icm
        }

        max_len = len("advantage_layer.0.weight")

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
        batch = self.buffer.sample_batch()
        states      = batch['obs'].to(DEVICE)
        next_states = batch['next_obs'].to(DEVICE)
        actions     = batch['acts'].to(DEVICE)
        rewards     = batch['rews'].to(DEVICE)
        dones       = batch['done'].to(DEVICE)

        # ----- ICM -----
        features        = self.online.feature_layer(states).detach()
        next_features   = self.online.feature_layer(next_states).detach()
        pred_action, pred_phi_next, phi_next = self.icm(features, next_features, actions)
        inverse_loss    = self.inverse_criterion(pred_action, actions)
        forward_loss    = self.forward_criterion(pred_phi_next, phi_next.detach())
        icm_loss        = (1 - ICM_BETA) * inverse_loss + ICM_BETA * forward_loss
        with torch.no_grad():
            intrinsic_reward = ICM_ETA * forward_loss.detach()

        # ----- DQN -----
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(1)
            q_next       = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_target     = rewards + intrinsic_reward + GAMMA_POW_N_STEP * q_next * (1 - dones)
        q_predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # td_value    = q_predicted - q_target.detach()
        dqn_loss    = self.dqn_criterion(q_predicted, q_target.detach())

        # ----- Update -----
        self.optimizer.zero_grad()
        dqn_loss.backward()
        U.clip_grad_norm_(self.online.parameters(), 5.0)
        # U.clip_grad_value_(self.online.parameters(), 1.0)

        self.optimizer.step()

        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        U.clip_grad_norm_(self.icm.parameters(), 5.0)
        # U.clip_grad_value_(self.icm.parameters(), 1.0)
        self.icm_optimizer.step()

        if self.frame_idx % TARGET_UPDATE_FRAMES == 0:
            if TARGET_UPDATE_TAU == 1.0:
                # Hard target update
                self.target.load_state_dict(self.online.state_dict())
            else:
                # Soft target update
                for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
                    target_param.data.copy_(TARGET_UPDATE_TAU * online_param.data + (1.0 - TARGET_UPDATE_TAU) * target_param.data)

        return dqn_loss.item(), forward_loss.item(), inverse_loss.item(), intrinsic_reward.mean().item()

    def save_model(self, path: str, dqn_only=False):
        # 儲存主 DQN 模型
        torch.save(self.online.state_dict(), path, pickle_protocol=4)
        if dqn_only:
            return
        assert path.endswith('.pth')
        # 儲存其他 metadata
        meta_path = path.replace('.pth', '.meta.pth')
        torch.save({
            'target'           : self.target.state_dict(),
            'icm'              : self.icm.state_dict(),
            'optimizer'        : self.optimizer.state_dict(),
            'icm_optimizer'    : self.icm_optimizer.state_dict(),
            'frame_idx'        : self.frame_idx,
            'rewards'          : self.rewards,
            'dqn_losses'       : self.dqn_losses,
            'forward_losses'   : self.forward_losses,
            'inverse_losses'   : self.inverse_losses,
            'intrinsic_rewards': self.intrinsic_rewards,
            'eval_rewards'     : self.eval_rewards,
            'best_eval_reward' : self.best_eval_reward,
            'buffer_state'     : self.buffer.get_state(),
        }, meta_path, pickle_protocol=4)

    def load_model(self, path, eval_mode=False):
        # 先載入 DQN 模型
        self.online.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
        if eval_mode:
            self.online.eval()
            return
        assert path.endswith('.pth')
        # 載入 metadata
        meta_path = path.replace('.pth', '.meta.pth')
        checkpoint = torch.load(meta_path, map_location=DEVICE, weights_only=False)
        self.target.load_state_dict(checkpoint['target'])
        self.icm.load_state_dict(checkpoint['icm'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.icm_optimizer.load_state_dict(checkpoint['icm_optimizer'])
        self.frame_idx         = checkpoint.get('frame_idx', 0)
        self.rewards           = checkpoint.get('rewards', [])
        self.dqn_losses        = checkpoint.get('dqn_losses', [])
        self.forward_losses    = checkpoint.get('forward_losses', [])
        self.inverse_losses    = checkpoint.get('inverse_losses', [])
        self.intrinsic_rewards = checkpoint.get('intrinsic_rewards', [])
        self.eval_rewards      = checkpoint.get('eval_rewards', [])
        self.best_eval_reward  = checkpoint.get('best_eval_reward', -np.inf)
        self.buffer.set_state(checkpoint['buffer_state'])


# ------- Training Functions -------

def plot_figure(agent: Agent, episode: int):
    plt.figure(figsize=(24, 10))

    plt.subplot(221)
    avg_reward = np.mean(agent.rewards[-PLOT_INTERVAL:]) if len(agent.rewards) >= PLOT_INTERVAL else np.mean(agent.rewards)
    plt.title(f"Life {episode} | Avg Reward {avg_reward:.1f}")
    plt.plot(1 + np.arange(len(agent.rewards)), agent.rewards, label='Reward')
    plt.xlim(left=1, right=len(agent.rewards))
    plt.ylim(bottom=max(-1000.0, min(agent.rewards)))
    plt.legend()

    arr = np.array(agent.rewards)
    # 補 0 讓長度是3的倍數
    if len(arr) % 3 != 0:
        arr = np.pad(arr, (0, 3 - len(arr) % 3))
    episode_rewards = arr.reshape(-1, 3).sum(axis=1)

    plt.subplot(223)
    avg_reward = np.mean(episode_rewards[-PLOT_INTERVAL//3:]) if len(episode_rewards) >= PLOT_INTERVAL // 3 else np.mean(episode_rewards)
    plt.title(f"Episode {episode // 3} | Avg Reward {avg_reward:.1f}")
    plt.plot(1 + np.arange(len(episode_rewards)), episode_rewards, label='Reward')
    plt.plot((1 + np.arange(len(agent.eval_rewards))) * EVAL_INTERVAL // 3, agent.eval_rewards, label='Eval Reward')
    plt.xlim(left=1, right=len(episode_rewards))
    plt.ylim(bottom=max(-1000.0, min(min(episode_rewards), min(agent.eval_rewards))))
    plt.legend()

    plt.subplot(222)
    plt.title("DQN Loss")
    plt.plot(agent.dqn_losses, label='DQN Loss')
    plt.xlim(left=0.0, right=len(agent.dqn_losses))
    plt.ylim(bottom=0.0,top=np.max(agent.dqn_losses[-int(len(agent.dqn_losses) // 2):] if len(agent.rewards) >= PLOT_INTERVAL else agent.dqn_losses))
    # plt.legend()

    plt.subplot(224)
    plt.title(f"ICM Loss | Intrinsic Reward = {ICM_ETA:.1f} x Forward Loss")
    plt.plot(agent.inverse_losses, label='Inverse Loss')
    plt.plot(agent.forward_losses, label='Forward Loss')
    plt.plot(agent.intrinsic_rewards, label='Intrinsic Reward')
    plt.xlim(left=0.0, right=len(agent.inverse_losses))
    plt.ylim(bottom=0.0)
    plt.legend()

    save_path = os.path.join(PLOT_DIR, f"episode_{episode}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

def evaluation(agent: Agent, episode: int, best_checkpoint_path='models/d3qn_per_bolzman_best.pth'):
    agent.online.eval()

    with torch.no_grad():
        eval_env = make_env(SKIP_FRAMES, STACK_FRAMES, life_episode=True, level=None)
        eval_rewards = [0, 0, 0]
        farest_x = 0
        for i in range(3):
            state = eval_env.reset()
            eval_reward = 0
            done = False
            while not done:
                e_action = agent.act(state, deterministic=True)
                state, reward, done, info = eval_env.step(e_action)
                farest_x = max(farest_x, info['x_pos'])
                if RENDER:
                    eval_env.render()
                eval_reward += reward
            eval_rewards[i] = eval_reward
        eval_env.close()

        total_eval_reward = sum(eval_rewards)
        agent.eval_rewards.append(total_eval_reward)
        if total_eval_reward > agent.best_eval_reward:
            agent.best_eval_reward = total_eval_reward

        print(
            f"Eval Reward: {eval_rewards[0]:.0f} + {eval_rewards[1]:.0f} + {eval_rewards[2]:.0f} = {total_eval_reward:.0f} | "
            f"Farest X {farest_x} | "
            f"Best Eval Reward: {agent.best_eval_reward:.0f}"
        )

        if total_eval_reward >= 3000 and total_eval_reward == agent.best_eval_reward:
            agent.save_model(best_checkpoint_path, dqn_only=True)
            print(f"Best model saved at episode {episode} with Eval Reward {total_eval_reward:.0f}")

    agent.online.train()

def train(
    agent: Agent,
    max_episodes: int,
    level: str=None,
    checkpoint_path='models/d3qn_icm_epsilonboltz.pth',
    best_checkpoint_path='models/d3qn_icm_epsilonboltz_best.pth',
):
    env = make_env(SKIP_FRAMES, STACK_FRAMES, life_episode=False, level=level)

    agent.online.train()

    # Warm-up
    state = env.reset()
    while agent.buffer.size < BATCH_SIZE:
        action = agent.act(state, deterministic=False)
        next_state, reward, done, _ = env.step(action)
        agent.buffer.store(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

    progress_bar = tqdm(total=10_000_000, desc="Training")
    progress_bar.update(len(agent.dqn_losses))

    start_episode = len(agent.rewards) + 1

    for episode in range(start_episode, max_episodes + 1):

        state = env.reset()
        episode_reward  = 0
        steps           = 0
        done            = False
        farest_x        = 0
        flag            = False

        while not done:
            agent.frame_idx += 1
            steps += 1

            action = agent.act(state, deterministic=(np.random.rand() >= EPSILON))

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            if RENDER:
                env.render()

            farest_x = max(farest_x, info['x_pos'])
            if info['flag_get']:
                flag = True

            agent.buffer.store(state, action, reward, next_state, done)
            state = next_state

            dqn_loss, forward_loss, inverse_loss, int_reward = agent.learn()
            agent.dqn_losses.append(dqn_loss)
            agent.forward_losses.append(forward_loss)
            agent.inverse_losses.append(inverse_loss)
            agent.intrinsic_rewards.append(int_reward)

            progress_bar.update(1)
            if agent.frame_idx >= 10_000_000:
                break

        agent.rewards.append(episode_reward)

        # Check parameters
        if CHECK_PARAM_INTERVAL > 0 and episode % CHECK_PARAM_INTERVAL == 0:
            agent.check_parameters()

        # Check gradients
        if CHECK_GRAD_INTERVAL > 0 and episode % CHECK_GRAD_INTERVAL == 0:
            agent.check_gradients()

        # Logging
        print(
            f"Episode {episode}\t| "
            f"Steps {steps}\t| "
            f"Reward {episode_reward:.0f}\t| "
            f"Flag {flag}\t| "
            f"Farest X {farest_x}"
        )

        # Evaluation
        if episode % EVAL_INTERVAL == 0:
            evaluation(agent, episode, best_checkpoint_path)

        # Plot
        if episode % PLOT_INTERVAL == 0:
            plot_figure(agent, episode)

        # Save model
        if episode % SAVE_INTERVAL == 0:
            avg_reward = np.mean(
                agent.rewards[-SAVE_INTERVAL:]
                if len(agent.rewards) >= SAVE_INTERVAL
                else agent.rewards
            )
            print(f"Avg Reward {avg_reward:.1f} | ")
            agent.save_model(checkpoint_path)
            print(f"Model saved at episode {episode}")

    env.close()
    progress_bar.close()

if __name__ == '__main__':
    checkpoint_path='models/d3qn_icm_lv1.pth'

    agent = Agent((4, 84, 84), 12)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    if os.path.isfile(checkpoint_path):
        agent.load_model(checkpoint_path)

    # train each level 3000 episodes
    train(agent, max_episodes=3000, level='1-1', checkpoint_path='models/d3qn_icm_lv1.pth')
    # train(agent, max_episodes=6000, level='1-2')
    # ep = 6000
    # for _ in range(10):
    #     ep += 300
    #     train(agent, max_episodes=ep, level='1-1')
    #     ep += 300
    #     train(agent, max_episodes=ep, level='1-2')
