import torch, os, numpy as np
from mario import RainbowDQNAgent, make_env
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# print(torch.cuda.is_available())

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = make_env(env)

agent = RainbowDQNAgent(
    env=env,
    memory_size=0,
    batch_size=128,
    target_update=10000,
    seed=-1,
)

agent.load_model("./models/Episode480.pth", eval_mode=True)
# agent.dqn.eval()

state = env.reset()
done = False
total_reward = 0
while not done:
    state = np.asarray(state)
    state, reward, done, info = env.step(agent.select_action(state))
    # print(info)
    env.render()
    total_reward += reward

env.close()

print(total_reward)
# print(env.observation_space.shape)