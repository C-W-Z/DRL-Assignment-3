import gym, pickle
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from student_agent_origin import Agent

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
    # if info["life"] < 2:  # 檢測第一條命死亡
    #     print(f"After death, x_pos: {info['x_pos']}")
    #     break
print(f"Total Reward: {total_reward}")
env.close()
