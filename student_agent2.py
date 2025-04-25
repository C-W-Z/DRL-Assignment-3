import gym
import numpy as np
import torch
import cv2
from collections import deque
import gym_super_mario_bros
from mario import RainbowDQNAgent, make_env, Frame_Processing
import pickle

class Agent(object):
    """Agent that uses a pre-trained Rainbow DQN model with skipframe and resize."""
    def __init__(self):
        # env = gym_super_mario_bros.make('SuperMarioBros-v0')
        # env = make_env(env)

        self.trajectory = []
        path = f"./human_play/play_1.pkl"
        with open(path, 'rb') as f:
            play = pickle.load(f)
            print(play['total_reward'])
            self.trajectory = play['trajectory']

        # Skipframe 參數
        self.skip_frames = 4
        self.frame_count = 0
        self.last_action = 0

        # 初始化 SkipAndMax 的緩衝區
        self.obs_buffer = deque(maxlen=2)  # 模擬 SkipAndMax

        # 初始化 BufferingWrapper 的緩衝區
        self.state_buffer = np.zeros((4, 84, 84), dtype=np.float32)  # 模擬 BufferingWrapper

    def get_state(self, processed_obs):
        """模擬 BufferingWrapper 的 observation 方法"""
        self.state_buffer[:-1] = self.state_buffer[1:]
        self.state_buffer[-1] = processed_obs  # processed_obs 形狀為 (1, 84, 84)，會自動調整為 (84, 84)
        return np.array(self.state_buffer).astype(np.float32) / 255.0  # 形狀：(4, 84, 84)

    def act(self, observation):
        """根據觀測值選擇動作"""

        self.obs_buffer.append(observation)
        self.frame_count += 1

        return self.trajectory[(self.frame_count) // self.skip_frames][1]

        # 如果是第一次調用
        if self.frame_count == 0:
            self.last_action = 0
            return 0 # NOOP action
            # self.obs_buffer.append(observation)
            # self.frame_count += 1

        # 模擬 SkipAndMax 的最大化操作
        max_frame = np.max(np.stack(self.obs_buffer), axis=0)  # 形狀：(240, 256, 3)

        # 使用 Frame_Processing 的 process 方法
        processed_obs = Frame_Processing.process(max_frame)  # 形狀：(1, 84, 84)

        # if self.frame_count == 100:
        #     cv2.imwrite("tmp.jpg", processed_obs[0])
        #     exit(0)

        # 實現 skipframe 邏輯
        if self.frame_count % self.skip_frames == 0:
            # 模擬 BufferingWrapper
            state = self.get_state(processed_obs)  # 形狀：(4, 84, 84)
            # self.last_action = self.agent.select_action(state)
            self.last_action = self.trajectory[self.frame_count // self.skip_frames][1]
            # print(self.last_action)
        # else:
        #     # 仍需更新 state_buffer 以保持狀態連續性
        #     self.get_state(processed_obs)

        return self.last_action