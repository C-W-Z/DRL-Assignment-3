import cv2
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
from torchvision import transforms as T
from collections import deque
from gym.wrappers import TimeLimit

class NoopResetEnv(gym.Wrapper):
    """
    From https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

class RandomStartEnv(gym.Wrapper):
    def __init__(self, env, random_steps=4):
        gym.Wrapper.__init__(self, env)
        self.random_steps = random_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.random_steps):
            obs, _, done, _ = self.env.step(np.random.randint(len(COMPLEX_MOVEMENT)))
            if done:
                obs = self.env.reset()
        return obs

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
        return self.env.reset()

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
            obs, _, done, _ = self.env.step(0)
            # if done:
            #     obs = self.env.reset()
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
        # frame = cv2.resize(frame, (84, 110), interpolation=cv2.INTER_AREA)
        # frame = frame[18:102, :]  # (84, 84)
        # 縮放到 (84, 84)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        # 轉為 (1, 84, 84)，規範化到 [0.0, 1.0]
        return frame.astype(np.float32)[np.newaxis, :, :] / 255.0
        # assert frame.shape == (1, 84, 84)

# class FrameStack(gym.Wrapper):
#     def __init__(self, env, n_steps=4):
#         super().__init__(env)
#         self.n_steps = n_steps
#         shp = env.observation_space.shape
#         self.observation_space = gym.spaces.Box(0, 1, shape=(n_steps * shp[0], shp[1], shp[2]), dtype=np.float32)
#         self.frames = np.zeros(self.observation_space.shape, dtype=np.float32)

#     def reset(self):
#         obs = self.env.reset()
#         obs = np.asarray(obs)
#         for i in range(self.n_steps):
#             self.frames[i] = obs
#         return self.frames

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         obs = np.asarray(obs)
#         self.frames[:-1] = self.frames[1:]
#         self.frames[-1] = obs
#         return self.frames, reward, done, info

class FrameStack(gym.Wrapper):
    """Stacks the last k observations along the channel dimension."""
    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 1, shape=(shp[0] * k, shp[1], shp[2]), dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Resets the environment and fills the frame stack with the initial observation."""
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return np.concatenate(self.frames, axis=0)

    def step(self, action: int):
        """Steps the environment and updates the frame stack."""
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.concatenate(self.frames, axis=0), reward, done, info

def make_env(skip_frames=4, stack_frames=4, life_episode=True, random_start=False, level: str=None):
    env = gym_super_mario_bros.make(f'SuperMarioBros-{level}-v0' if level else 'SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    # env = NoopResetEnv(env)
    if random_start:
        env = RandomStartEnv(env, random_steps=4)
    env = SkipAndMax(env, skip=skip_frames)
    if life_episode:
        env = LifeEpisode(env)
    env = FrameProcessing(env) # (1, 84, 84) [0.0, 1.0]
    env = FrameStack(env, k=stack_frames) # (4, 84, 84)
    # env = TimeLimit(env, 3000)
    return env

if __name__ == "__main__":
    env = make_env(life_episode=False, level=None)
    print(f"observation_space.shape: {env.observation_space.shape}")
    obs = env.reset()
    assert obs.shape == env.observation_space.shape
    cv2.imwrite("test2.jpg", (obs[0] * 255).astype(np.uint8))
