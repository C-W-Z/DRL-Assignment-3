import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from torch.optim import Adam
from tqdm import tqdm
from numba import njit
from env_wrapper import make_env
from per import PrioritizedReplayBuffer

# -----------------------------
# Hyperparameters
# -----------------------------
RENDER                  = True
MAX_TRAINING_FRAMES     = 2000000

# Env Wrappers
SKIP_FRAMES             = 4
STACK_FRAMES            = 4

# DQN
TARGET_UPDATE           = 10
TAU                     = 0.005
LEARNING_RATE           = 5e-4
ADAM_EPS                = 0.00015

# Epsilon Boltzmann Exploration
EPSILON_START           = 1.0
EPSILON_MIN             = 0.1
EPSILON_DECAY           = 0.999
EXPLORE_TAU             = 1.0

# Prioritized Replay Buffer
MEMORY_SIZE             = 50000
BATCH_SIZE              = 64
GAMMA                   = 0.95
N_STEP                  = 5
ALPHA                   = 0.6
BETA_START              = 0.4
BETA_FRAMES             = 1000000
PRIOR_EPS               = 1e-6
GAMMA_POW_N_STEP = GAMMA ** N_STEP

# Output
EVAL_INTERVAL           = 30
SAVE_INTERVAL           = 150
PLOT_INTERVAL           = 30
CHECK_PARAM_INTERVAL    = 150
CHECK_GRAD_INTERVAL     = 150
MODEL_DIR               = "./models"
PLOT_DIR                = "./plots"

# -----------------------------
# Double Dueling Deep Recurrent Q Network
# -----------------------------
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

# -----------------------------
# Prioritized Replay Buffer
# -----------------------------
@njit
def _get_beta_by_frame(frame_idx):
    return min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

# -----------------------------
# Agent
# -----------------------------
class Agent:
    def __init__(self, obs_shape: Tuple, n_actions: int, device=None):
        self.device     = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions  = n_actions

        self.online     = D3QN(obs_shape[0], n_actions).to(self.device)
        self.target     = D3QN(obs_shape[0], n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer  = Adam(self.online.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

        self.buffer     = PrioritizedReplayBuffer(obs_shape, MEMORY_SIZE, BATCH_SIZE, ALPHA, N_STEP, GAMMA, PRIOR_EPS)

        self.criterion  = nn.MSELoss(reduction='none')

        self.epsilon           = EPSILON_START
        self.frame_idx         = 0
        self.rewards           = []
        self.dqn_losses        = []
        self.eval_rewards      = []
        self.best_eval_reward  = -np.inf

    def act(self, state, deterministic=False, tau=EXPLORE_TAU):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
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
        batch       = self.buffer.sample_batch(_get_beta_by_frame(self.frame_idx))
        states      = torch.tensor(batch['obs'], dtype=torch.float32, device=self.device)
        actions     = torch.tensor(batch['acts'], dtype=torch.int64, device=self.device)
        rewards     = torch.tensor(batch['rews'], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch['next_obs'], dtype=torch.float32, device=self.device)
        dones       = torch.tensor(batch['done'], dtype=torch.float32, device=self.device)
        weights     = torch.tensor(batch['weights'], dtype=torch.float32, device=self.device)
        indices     = batch['indices']

        # ------- DQN --------
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(1)
            q_next       = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_target     = rewards + GAMMA_POW_N_STEP * q_next * (1 - dones)
        q_predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_value    = q_predicted - q_target.detach()
        dqn_loss    = (self.criterion(q_predicted, q_target.detach()) * weights).mean()

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
        self.dqn_losses        = checkpoint.get('dqn_losses', [])
        self.eval_rewards      = checkpoint.get('eval_rewards', [])
        self.best_eval_reward  = checkpoint.get('best_eval_reward', -np.inf)
        self.buffer.set_state(checkpoint['buffer_state'])

# -----------------------------
# Training Loop
# -----------------------------
def plot_figure(agent: Agent, episode: int):
    plt.figure(figsize=(20, 15))

    plt.subplot(311)
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

    plt.subplot(312)
    avg_reward = np.mean(episode_rewards[-PLOT_INTERVAL//3:]) if len(episode_rewards) >= PLOT_INTERVAL // 3 else np.mean(episode_rewards)
    plt.title(f"Episode {episode // 3} | Avg Reward {avg_reward:.1f}")
    plt.plot(1 + np.arange(len(episode_rewards)), episode_rewards, label='Reward')
    plt.plot((1 + np.arange(len(agent.eval_rewards))) * EVAL_INTERVAL // 3, agent.eval_rewards, label='Eval Reward')
    plt.xlim(left=1, right=len(episode_rewards))
    plt.ylim(bottom=max(-1000.0, min(min(episode_rewards), min(agent.eval_rewards))))
    plt.legend()

    plt.subplot(313)
    plt.title("DQN Loss")
    plt.plot(agent.dqn_losses, label='DQN Loss')
    plt.xlim(left=0.0, right=len(agent.dqn_losses))
    plt.ylim(bottom=0.0,top=np.max(agent.dqn_losses[-int(len(agent.dqn_losses) // 2):] if len(agent.rewards) >= PLOT_INTERVAL else agent.dqn_losses))
    # plt.legend()

    save_path = os.path.join(PLOT_DIR, f"episode_{episode}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    tqdm.write(f"Plot saved to {save_path}")

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

        tqdm.write(
            f"Eval Reward: {eval_rewards[0]:.0f} + {eval_rewards[1]:.0f} + {eval_rewards[2]:.0f} = {total_eval_reward:.0f} | "
            f"Farest X {farest_x} | "
            f"Best Eval Reward: {agent.best_eval_reward:.0f}"
        )

        if total_eval_reward >= 3000 and total_eval_reward == agent.best_eval_reward:
            agent.save_model(best_checkpoint_path)
            tqdm.write(f"Best model saved at episode {episode} with Eval Reward {total_eval_reward:.0f}")

    agent.online.train()

def train(num_episodes: int, checkpoint_path='models/d3qn_per_bolzman.pth', best_checkpoint_path='models/d3qn_per_bolzman_best.pth'):
    env = make_env(SKIP_FRAMES, STACK_FRAMES, False)
    print(f"observation_space.shape: {env.observation_space.shape}")

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
        action = np.random.randint(env.action_space.n)
        next_state, reward, done, _ = env.step(action)
        agent.buffer.store(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

    # Training
    progress_bar = tqdm(total=MAX_TRAINING_FRAMES, desc="Training")
    progress_bar.update(len(agent.dqn_losses))

    for episode in range(start_episode, num_episodes + 1):

        state = env.reset()
        episode_reward  = 0
        steps           = 0
        done            = False
        farest_x        = 0
        flag            = False

        while not done:
            agent.frame_idx += 1
            steps += 1

            action = agent.act(state, deterministic=(np.random.rand() >= agent.epsilon))

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            if RENDER:
                env.render()

            farest_x = max(farest_x, info['x_pos'])
            if info['flag_get']:
                flag = True

            agent.buffer.store(state, action, reward, next_state, done)
            state = next_state

            dqn_loss = agent.learn()
            agent.dqn_losses.append(dqn_loss)

            progress_bar.update(1)
            if agent.frame_idx >= MAX_TRAINING_FRAMES:
                break

        agent.rewards.append(episode_reward)

        # Check parameters
        if CHECK_PARAM_INTERVAL > 0 and episode % CHECK_PARAM_INTERVAL == 0:
            agent.check_parameters()

        # Check gradients
        if CHECK_GRAD_INTERVAL > 0 and episode % CHECK_GRAD_INTERVAL == 0:
            agent.check_gradients()

        # Logging
        tqdm.write(
            f"Episode {episode}\t| "
            f"Steps {steps}\t| "
            f"Reward {episode_reward:.0f}\t| "
            f"Flag {flag}\t| "
            f"Epsilon {agent.epsilon:.4f} | "
            f"Farest X {farest_x}"
        )

        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

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
            tqdm.write(f"Avg Reward {avg_reward:.1f} | ")
            agent.save_model(checkpoint_path)
            tqdm.write(f"Model saved at episode {episode}")

        if agent.frame_idx >= MAX_TRAINING_FRAMES:
            break

    progress_bar.close()
    env.close()

if __name__ == '__main__':
    train(num_episodes=20000)
