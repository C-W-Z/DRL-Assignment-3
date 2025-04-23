import gym
import numpy as np
import torch
import cv2
from collections import deque
import gym_super_mario_bros
from mario import RainbowDQNAgent, make_env

class Agent(object):
    """Agent that uses a pre-trained Rainbow DQN model with skipframe and resize."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = make_env(env)

        # 初始化 RainbowDQNAgent
        obs_shape = (4, 84, 84)  # 預處理後的形狀：堆疊 4 幀，每幀 84x84
        self.agent = RainbowDQNAgent(
            env=env,
            memory_size=0,
            batch_size=128,
            target_update=10000,
            seed=777,
            gamma=0.99,
            alpha=0.6,
            beta=0.4,
            prior_eps=1e-6,
            v_min=-500.0,
            v_max=5000.0,
            atom_size=51,
            n_step=3,
            tau=0.9,
            lr=0.0001,
            model_save_dir="./models",
        )
        # 設置設備
        self.device = self.agent.device

        # 載入預訓練模型
        self.agent.load_model("Best_Episode180.pth")  # 假設你的最佳模型保存在此路徑
        self.agent.dqn.eval()  # 設置為評估模式

        # Skipframe 參數
        self.skip_frames = 4  # 每 4 幀選擇一次動作
        self.frame_count = 0
        self.last_action = 0  # 保存上一次選擇的動作

        # 狀態堆疊緩衝區
        self.state_buffer = deque(maxlen=4)  # 儲存 4 幀的觀測值
        # 初始化緩衝區（假設初始觀測值為 0）
        dummy_obs = np.zeros((84, 84, 1), dtype=np.float32)
        for _ in range(4):
            self.state_buffer.append(dummy_obs)

    def process_observation(self, observation):
        """處理觀測值：調整大小並轉為灰度圖"""
        # 假設 observation 是 (240, 256, 3) 的 RGB 圖像
        if observation.shape == (240, 256, 3):
            # 轉為灰度圖：使用加權平均法
            img = observation.astype(np.float32)
            img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
            # 調整大小到 84x110
            resized = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
            # 裁剪到 84x84
            resized = resized[18:102, :]
            # 增加一個通道維度：(84, 84, 1)
            resized = np.reshape(resized, (84, 84, 1))
            # 標準化到 [0, 1]
            processed = resized.astype(np.float32) / 255.0
            return processed
        else:
            raise ValueError(f"Unexpected observation shape: {observation.shape}")

    def get_state(self):
        """獲取堆疊的狀態：(4, 84, 84)"""
        state = np.stack(self.state_buffer, axis=0)  # 形狀：(4, 84, 84, 1)
        state = np.moveaxis(state, 3, 1)  # 轉為 (4, 1, 84, 84)
        return state.squeeze(1)  # 移除多餘的維度，得到 (4, 84, 84)

    def act(self, observation):
        """根據觀測值選擇動作，包含 skipframe 和 resize 處理"""
        # 處理觀測值（resize 和灰度處理）
        processed_obs = self.process_observation(observation)

        # 將處理後的觀測值加入緩衝區
        self.state_buffer.append(processed_obs)

        # 實現 skipframe 邏輯
        self.frame_count += 1
        if self.frame_count % self.skip_frames == 0:
            # 每 skip_frames 幀選擇一次新動作
            # 獲取堆疊的狀態
            state = self.get_state()
            # 使用 RainbowDQNAgent 的 select_action 方法選擇動作
            self.last_action = self.agent.select_action(state)
        # 返回動作（重複使用上一次的動作，直到下一次選擇）
        return self.last_action
