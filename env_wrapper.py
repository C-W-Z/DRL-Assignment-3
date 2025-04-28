import cv2
import gym
from gym.wrappers import TimeLimit
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np

class SkipAndMax(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer[i & 1] = np.asarray(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(self._obs_buffer, axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = np.asarray(obs)
        self._obs_buffer[0] = self._obs_buffer[1] = obs
        return obs

class LifeEpisode(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 2
        self.real_done = True
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.real_done = done
        lives = info.get('life', getattr(self.env.unwrapped, '_life', 0))
        if lives < self.lives:
            done = True
        self.lives = lives
        return obs, reward, done, info
    def reset(self):
        if self.real_done:
            obs = self.env.reset()
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = getattr(self.env.unwrapped, '_life', 2)
        return obs

class FrameProcessing(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, 84, 84), dtype=np.float32)

    def observation(self, obs):
        return FrameProcessing.process(obs)

    @staticmethod
    def process(frame):
        frame = np.asarray(frame)
        # assert frame.shape == (240, 256, 3), "Wrong resolution."
        # 使用 OpenCV 轉灰度
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # (240, 256), uint8
        # 縮放到 (110, 84)，然後裁剪
        frame = cv2.resize(frame, (84, 110), interpolation=cv2.INTER_AREA)
        frame = frame[18:102, :]  # (84, 84)
        # 轉為 (1, 84, 84)，規範化到 [0.0, 1.0]
        return frame.astype(np.float32)[np.newaxis, :, :] / 255.0
        # assert frame.shape == (1, 84, 84)

class FrameStack(gym.Wrapper):
    def __init__(self, env, n_steps=4):
        super().__init__(env)
        self.n_steps = n_steps
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 1, shape=(shp[0] * n_steps, shp[1], shp[2]), dtype=np.float32)
        self.frames = np.zeros(self.observation_space.shape, dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        obs = np.asarray(obs)
        for i in range(self.n_steps):
            self.frames[i] = obs
        return self.frames

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.asarray(obs)
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = obs
        return self.frames, reward, done, info

def make_env(skip_frames=4, stack_frames=4, max_episode_steps=3000):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipAndMax(env, skip=skip_frames)
    env = LifeEpisode(env)
    env = FrameProcessing(env)
    env = FrameStack(env, n_steps=stack_frames)
    # env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
