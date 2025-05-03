from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from env_wrapper import make_env, SkipAndMax, FrameProcessing, FrameStack
from train import Agent as DQNAgent

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

env = SkipAndMax(env, skip=4)
env = FrameProcessing(env)
env = FrameStack(env, k=4)

# env = make_env(life_episode=False, random_start=False, level=None)

agent = DQNAgent(env.observation_space.shape, env.action_space.n)
# agent.load_model("./models/d3qn_per_bolzman.pth", eval_mode=True)
agent.load_model("./models/d3qn_icm_best_8636score.pth", eval_mode=True)

while True:
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state, reward, done, info = env.step(agent.act(state))
        # print(info)
        env.render()
        total_reward += reward

    print(total_reward)

env.close()
