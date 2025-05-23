import os
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from torch.optim import Adam
from numba import njit
from env_wrapper import make_env
from per import PrioritizedReplayBuffer
from tqdm import tqdm

# -----------------------------
# Hyperparameters
# -----------------------------
RENDER                  = False
MAX_FRAMES              = 10_000_000

# Env Wrappers
SKIP_FRAMES             = 4
STACK_FRAMES            = 4

# DQN
TARGET_UPDATE_FRAMES    = 500
TARGET_UPDATE_TAU       = 0.1
DQN_LEARNING_RATE       = 2.5e-4
DQN_ADAM_EPS            = 1.5e-4
DQN_WEIGHT_DECAY        = 1e-6

# Intrinsic Curiosity Module
ICM_BETA                = 0.2
ICM_ETA                 = 1.0
ICM_EMBED_DIM           = 256
ICM_LEARNING_RATE       = 2.5e-4

# Epsilon Boltzmann Exploration
EPSILON                 = 0.1
EXPLORE_TAU             = 1.0

# Prioritized Replay Buffer
MEMORY_SIZE             = 30_000
BATCH_SIZE              = 64
GAMMA                   = 0.9 # 重生點的存在導致累積reward不是完全連續的，因此降低gamma
N_STEP                  = 5
ALPHA                   = 0.6
BETA_START              = 0.4
BETA_FRAMES             = 2_000_000
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

# ------- Prioritized Replay Buffer -------

@njit
def _get_beta_by_frame(frame_idx: int):
    return min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

# ------- Intrinsic Curiosity Module -------

class ICM(nn.Module):
    def __init__(self, feature_dimension: int, n_actions: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dimension, ICM_EMBED_DIM),
            nn.ReLU(),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(ICM_EMBED_DIM * 2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, n_actions)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(ICM_EMBED_DIM + n_actions, 512),
            nn.LeakyReLU(),
            nn.Linear(512, ICM_EMBED_DIM)
        )

    def forward(self, features, next_features, actions):
        feature              = self.encoder(features)
        next_feature         = self.encoder(next_features)

        inverse_input        = torch.cat([feature, next_feature], dim=1)
        predict_action       = self.inverse_model(inverse_input)

        action_onehot        = F.one_hot(actions, num_classes=predict_action.size(-1)).float()
        forward_input        = torch.cat([feature, action_onehot], dim=1)
        predict_next_feature = self.forward_model(forward_input)

        return predict_action, predict_next_feature, next_feature

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

        # 用 Xavier 初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_uniform_(m.weight)
        #         m.bias.data.fill_(0.01)

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
    def __init__(self, obs_shape: Tuple, n_actions: int, device=None):
        self.device         = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions      = n_actions

        self.online         = D3QN(obs_shape[0], n_actions).to(self.device)
        self.target         = D3QN(obs_shape[0], n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.icm            = ICM(self.online.feature_dimension, n_actions).to(self.device)

        # self.optimizer      = Adam(self.online.parameters(), lr=DQN_LEARNING_RATE, eps=DQN_ADAM_EPS, weight_decay=DQN_WEIGHT_DECAY)
        self.optimizer      = build_dqn_optimizer(self.online)
        self.icm_optimizer  = Adam(self.icm.parameters(), lr=ICM_LEARNING_RATE)

        self.buffer         = PrioritizedReplayBuffer(obs_shape, MEMORY_SIZE, BATCH_SIZE, ALPHA, N_STEP, GAMMA, PRIOR_EPS)

        self.dqn_criterion      = nn.MSELoss(reduction='none')
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
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online(state_tensor)  # Shape: (1, n_actions)
            if deterministic:
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
            tqdm.write(f"\n[{model_name}] Parameter Statistics at Frame {self.frame_idx}:")
            for stat in param_stats:
                tqdm.write(stat)

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
            tqdm.write(f"\n[{model_name}] Gradient Statistics at Frame {self.frame_idx}:")
            for stat in grad_stats:
                tqdm.write(stat)

    def learn(self):
        batch       = self.buffer.sample_batch(_get_beta_by_frame(self.frame_idx))
        states      = torch.tensor(batch['obs'], dtype=torch.float32, device=self.device)
        actions     = torch.tensor(batch['acts'], dtype=torch.int64, device=self.device)
        rewards     = torch.tensor(batch['rews'], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch['next_obs'], dtype=torch.float32, device=self.device)
        dones       = torch.tensor(batch['done'], dtype=torch.float32, device=self.device)
        weights     = torch.tensor(batch['weights'], dtype=torch.float32, device=self.device)
        indices     = batch['indices']

        # ----- ICM -----
        features        = self.online.feature_layer(states).detach()
        next_features   = self.online.feature_layer(next_states).detach()
        predict_action, predict_next_feature, next_feature = self.icm(features, next_features, actions)
        inverse_loss    = self.inverse_criterion(predict_action, actions)
        forward_loss    = self.forward_criterion(predict_next_feature, next_feature)
        icm_loss        = (1 - ICM_BETA) * inverse_loss + ICM_BETA * forward_loss
        with torch.no_grad():
            intrinsic_reward = ICM_ETA * forward_loss.detach()

        # ----- DQN -----
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(1)
            q_next       = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_target     = rewards + intrinsic_reward + GAMMA_POW_N_STEP * q_next * (1 - dones)
        q_predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_value    = q_predicted - q_target.detach()
        dqn_loss    = (self.dqn_criterion(q_predicted, q_target.detach()) * weights).mean()

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

        self.buffer.update_priorities(indices, td_value.detach().abs().cpu().numpy())

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

    def load_model(self, path, eval_mode=False, load_memory=True):
        # 先載入 DQN 模型
        self.online.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        if eval_mode:
            self.online.eval()
            return
        assert path.endswith('.pth')
        # 載入 metadata
        meta_path = path.replace('.pth', '.meta.pth')
        checkpoint = torch.load(meta_path, map_location=self.device, weights_only=False)
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
        if load_memory:
            self.buffer.set_state(checkpoint['buffer_state'])

# ------- Training Functions -------

def plot_figure(agent: Agent, episode: int):
    plt.figure(figsize=(20, 15))

    plt.subplot(311)
    avg_reward = np.mean(agent.rewards[-PLOT_INTERVAL:]) if len(agent.rewards) >= PLOT_INTERVAL else np.mean(agent.rewards)
    plt.title(f"Life {episode} | Avg Reward {avg_reward:.1f}")
    plt.plot(1 + np.arange(len(agent.rewards)), agent.rewards, label='Reward')
    plt.plot((1 + np.arange(len(agent.eval_rewards))) * EVAL_INTERVAL, agent.eval_rewards, label='Eval Reward')
    plt.xlim(left=1, right=len(agent.rewards))
    plt.ylim(bottom=max(-1000.0, min(agent.rewards)))
    plt.legend()

    plt.subplot(312)
    plt.title("DQN Loss")
    plt.plot(agent.dqn_losses, label='DQN Loss')
    plt.xlim(left=0.0, right=len(agent.dqn_losses))
    plt.ylim(bottom=0.0,top=min(50.0, np.max(agent.dqn_losses)))
    # plt.legend()

    plt.subplot(313)
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
    tqdm.write(f"Plot saved to {save_path}")

def evaluation(agent: Agent, episode: int, best_checkpoint_path='models/d3qn_per_bolzman_best.pth'):
    agent.online.eval()

    with torch.no_grad():
        eval_env = make_env(SKIP_FRAMES, STACK_FRAMES, life_episode=True, random_start=False, level=None)
        eval_rewards = [0, 0, 0]
        farest_x = 0
        prev_stage = 1
        for i in range(3):
            state = eval_env.reset()
            eval_reward = 0
            done = False
            while not done:
                e_action = agent.act(state, deterministic=True)
                state, reward, done, info = eval_env.step(e_action)
                stage = info['stage']
                if prev_stage < stage:
                    farest_x = info['x_pos']
                else:
                    farest_x = max(farest_x, info['x_pos'])
                prev_stage = stage
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
            f"Stage {prev_stage} | "
            f"Farest X {farest_x} | "
            f"Best Eval Reward: {agent.best_eval_reward:.0f}"
        )

        if total_eval_reward >= 3000 and total_eval_reward == agent.best_eval_reward:
            agent.save_model(best_checkpoint_path, dqn_only=True)
            tqdm.write(f"Best model saved at episode {episode} with Eval Reward {total_eval_reward:.0f}")

    agent.online.train()

def train(
    agent: Agent,
    max_episodes: int,
    level: str=None,
    checkpoint_path='models/d3qn_icm_epsilonboltz.pth',
    best_checkpoint_path='models/d3qn_icm_epsilonboltz_best.pth',
):
    env = make_env(SKIP_FRAMES, STACK_FRAMES, life_episode=False, random_start=True, level=level)

    agent.online.train()

    progress_bar = tqdm(total=MAX_FRAMES, desc="Training")
    progress_bar.update(agent.frame_idx)

    # Warm-up
    state = env.reset()
    while agent.buffer.size < BATCH_SIZE:
        action = agent.act(state, deterministic=False)
        next_state, reward, done, _ = env.step(action)
        agent.buffer.store(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
        progress_bar.update(1)

    start_episode = len(agent.rewards) + 1

    for episode in range(start_episode, max_episodes + 1):

        state = env.reset()
        episode_reward  = 0
        steps           = 0
        done            = False
        farest_x        = 0
        prev_stage      = 1
        prev_life       = 2

        while not done:
            agent.frame_idx += 1
            steps += 1

            action = agent.act(state, deterministic=(np.random.rand() >= EPSILON))

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            if RENDER:
                env.render()

            x_pos        = info['x_pos']
            stage        = info['stage']
            life         = info['life']

            if prev_stage < stage:
                farest_x = x_pos
            else:
                farest_x = max(farest_x, x_pos)
            # if life < prev_life:
            #     done     = True

            prev_stage   = stage
            prev_life    = life

            agent.buffer.store(state, action, reward, next_state, done)
            state = next_state

            dqn_loss, forward_loss, inverse_loss, int_reward = agent.learn()
            agent.dqn_losses.append(dqn_loss)
            agent.forward_losses.append(forward_loss)
            agent.inverse_losses.append(inverse_loss)
            agent.intrinsic_rewards.append(int_reward)

            progress_bar.update(1)
            if agent.frame_idx >= MAX_FRAMES:
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
            f"Stage {prev_stage} | "
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
            tqdm.write(f"Avg Reward {avg_reward:.1f}")
            agent.save_model(checkpoint_path)
            tqdm.write(f"Model saved at episode {episode}")

        if agent.frame_idx >= MAX_FRAMES:
                break

    env.close()
    progress_bar.close()

if __name__ == '__main__':
    checkpoint_path='models/d3qn_icm_1200.pth'

    agent = Agent((4, 84, 84), 12)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    if os.path.isfile(checkpoint_path):
        agent.load_model(checkpoint_path)

    # 輪流練

    if len(agent.rewards) < 200:
        train(agent, max_episodes=200, level=None, checkpoint_path='models/d3qn_icm_200.pth', best_checkpoint_path='models/d3qn_icm_best.pth')

    if len(agent.rewards) < 400:
        train(agent, max_episodes=400, level='1-2', checkpoint_path='models/d3qn_icm_400.pth', best_checkpoint_path='models/d3qn_icm_best.pth')

    if len(agent.rewards) < 500:
        train(agent, max_episodes=500, level=None, checkpoint_path='models/d3qn_icm_500.pth', best_checkpoint_path='models/d3qn_icm_best.pth')

    if len(agent.rewards) < 600:
        train(agent, max_episodes=600, level='1-2', checkpoint_path='models/d3qn_icm_600.pth', best_checkpoint_path='models/d3qn_icm_best.pth')

    if len(agent.rewards) < 800:
        train(agent, max_episodes=800, level=None, checkpoint_path='models/d3qn_icm_800.pth', best_checkpoint_path='models/d3qn_icm_best.pth')

    if len(agent.rewards) < 900:
        train(agent, max_episodes=900, level='1-1', checkpoint_path='models/d3qn_icm_900.pth', best_checkpoint_path='models/d3qn_icm_best.pth')

    if len(agent.rewards) < 1300:
        train(agent, max_episodes=1300, level=None, checkpoint_path='models/d3qn_icm_1300.pth', best_checkpoint_path='models/d3qn_icm_best.pth')

    if len(agent.rewards) < 1500:
        train(agent, max_episodes=1500, level='1-2', checkpoint_path='models/d3qn_icm_1500.pth', best_checkpoint_path='models/d3qn_icm_best.pth')

    # if len(agent.rewards) < 2000:
    #     train(agent, max_episodes=2000, level='1-3', checkpoint_path='models/d3qn_icm_2000.pth', best_checkpoint_path='models/d3qn_icm_best.pth')

    train(agent, max_episodes=10000, level=None, checkpoint_path='models/d3qn_icm.pth', best_checkpoint_path='models/d3qn_icm_best.pth')

    # stages = ['1-1', '1-2', None]

    # e = 0
    # for _ in range(10):
    #     for s in stages:
    #         e += 600
    #         train(
    #             agent,
    #             max_episodes=e,
    #             level=s,
    #             checkpoint_path=f'models/d3qn_icm_{e}.pth',
    #             best_checkpoint_path='models/d3qn_icm_best.pth'
    #         )
