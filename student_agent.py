import numpy as np
import torch
from collections import deque
from train import Agent as DQNAgent, SKIP_FRAMES, STACK_FRAMES
from env_wrapper import FrameProcessing

class Agent(object):
    def __init__(self):
        self.agent = DQNAgent((4, 84, 84), 12)
        self.agent.load_model("./models/d3qn_icm_best_5802score.pth", eval_mode=True)
        self.agent.online = torch.jit.script(self.agent.online) # 加速推理

        # Skipframe 參數
        self.last_action  = 0
        self.frame_count  = 0

        # 初始化 SkipAndMax 的緩衝區
        self.obs_buffer = deque(maxlen=2)

        # 初始化 BufferingWrapper 的緩衝區
        self.frames = deque(maxlen=STACK_FRAMES)

    def act(self, observation):
        observation = np.asarray(observation)

        self.obs_buffer.append(observation)
        while len(self.obs_buffer) < 2:
            self.obs_buffer.append(observation)

        if self.frame_count % SKIP_FRAMES == 0:
            max_frame = np.max(np.stack(self.obs_buffer), axis=0)   # (240, 256, 3) [0, 255]
            processed_obs = FrameProcessing.process(max_frame)      # (1, 84, 84), [0.0, 1.0]

            self.frames.append(processed_obs)
            while len(self.frames) < STACK_FRAMES:
                self.frames.append(processed_obs)

            state = np.concatenate(self.frames, axis=0)  # (4, 84, 84)
            self.last_action = self.agent.act(state)

        self.frame_count += 1
        return self.last_action

if __name__ == "__main__":
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    # 創建環境
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    # 初始化 Agent
    agent = Agent()

    # 運行測試
    observation = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        env.render()  # 可視化
        print(f"Total Reward: {total_reward}")
        env.close()
