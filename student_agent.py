import numpy as np
import torch
from collections import deque
from train import Agent as DQNAgent, SKIP_FRAMES, STACK_FRAMES
from env_wrapper import FrameProcessing

class Agent(object):
    def __init__(self):
        self.agent = DQNAgent((4, 84, 84), 12)
        self.agent.load_model("./d3qn_icm_best_8636score.pth", eval_mode=True)
        self.agent.online = torch.jit.script(self.agent.online) # 加速推理

        # Skipframe 參數
        self.last_action  = 0
        self.frame_count  = 0

        # 初始化 SkipAndMax 的緩衝區
        self.obs_buffer = np.zeros((2,) + (240, 256, 3), dtype=np.uint8)

        # 初始化 BufferingWrapper 的緩衝區
        self.frames = deque(maxlen=STACK_FRAMES)

    def act(self, observation):
        observation = np.asarray(observation)

        if self.frame_count == 0:
            processed_obs = FrameProcessing.process(observation) # (1, 84, 84), [0.0, 1.0]

            for _ in range(STACK_FRAMES):
                self.frames.append(processed_obs)

            state = np.concatenate(self.frames, axis=0)  # (4, 84, 84)
            self.last_action = self.agent.act(state)

            self.frame_count += 1
            return self.last_action

        self.obs_buffer[self.frame_count & 1] = observation

        if self.frame_count % SKIP_FRAMES == 0:
            max_frame = np.max(self.obs_buffer, axis=0)        # (240, 256, 3) [0, 255]
            processed_obs = FrameProcessing.process(max_frame) # (1, 84, 84), [0.0, 1.0]

            self.frames.append(processed_obs)

            state = np.concatenate(self.frames, axis=0)  # (4, 84, 84)
            self.last_action = self.agent.act(state, deterministic=(np.random.rand() >= 0.1))

        self.frame_count += 1
        return self.last_action

if __name__ == "__main__":
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    avg_reward = 0
    for _ in range(10):

        # 創建環境
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)

        # 初始化 Agent
        agent = Agent()

        # 運行測試
        observation = env.reset()
        total_reward = 0
        done = False
        # for _ in range(4):
        #     observation, reward, done, info = env.step(np.random.randint(len(COMPLEX_MOVEMENT)))
        while not done:
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            env.render()  # 可視化
        print(f"Total Reward: {total_reward}")
        env.close()

        avg_reward += total_reward
    avg_reward /= 10
    print("Avg reward:",avg_reward)
