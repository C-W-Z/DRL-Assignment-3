import torch, os, numpy as np, gym
from mario import RainbowDQNAgent, make_env
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from torchvision import transforms as T

# print(torch.cuda.is_available())

class GrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.transform = T.Compose([
            T.ToPILImage(), T.Grayscale(), T.Resize((84,90)), T.ToTensor()
        ])
        self.observation_space = gym.spaces.Box(0.0,1.0,shape=(1,84,90),dtype=np.float32)
    def observation(self, obs):
        return self.transform(obs)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
# env = make_env(env)
env = GrayScaleResize(env)

# agent = RainbowDQNAgent(
#     env=env,
#     memory_size=0,
#     batch_size=128,
#     target_update=10000,
#     seed=-1,
# )

# agent.load_model("./models/Episode480.pth", eval_mode=True)
# agent.dqn.eval()

state = env.reset()
print(state)
exit(0)
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