import pygame
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
from collections import deque
import cv2
from gym import spaces
import pickle

# 環境預處理 Wrapper
class SkipAndMax(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipAndMax, self).__init__(env)
        self.obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        origin_reward = 0
        raw_states = []  # 儲存原始state用於渲染
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            raw_states.append(obs)  # 保存原始state
            origin_reward += info.get('reward', 0)
            self.obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        info['reward'] = origin_reward
        info['raw_states'] = raw_states  # 將原始state存入info
        return max_frame, total_reward, done, info

    def reset(self):
        self.obs_buffer.clear()
        obs = self.env.reset()
        self.obs_buffer.append(obs)
        return obs

class Frame_Processing(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(Frame_Processing, self).__init__(env)
        old_shape = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8).shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                            dtype=np.float32)

    def observation(self, obs):
        return Frame_Processing.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Wrong resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1]).astype(np.uint8)
        return np.moveaxis(x_t, 2, 0)

class BufferingWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferingWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                            old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return np.array(self.buffer).astype(np.float32) / 255.0

# InputHandler（保持不變）
class InputHandler:
    def __init__(self):
        self.key_mapping = {
            pygame.K_n: ['NOOP'],
            pygame.K_RIGHT: ['right'],
            pygame.K_SPACE: ['A'],
            pygame.K_LSHIFT: ['B'],
            pygame.K_LEFT: ['left'],
            pygame.K_DOWN: ['down'],
            pygame.K_UP: ['up'],
            (pygame.K_RIGHT, pygame.K_SPACE): ['right', 'A'],
            (pygame.K_RIGHT, pygame.K_LSHIFT): ['right', 'B'],
            (pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT): ['right', 'A', 'B'],
            (pygame.K_LEFT, pygame.K_SPACE): ['left', 'A'],
            (pygame.K_LEFT, pygame.K_LSHIFT): ['left', 'B'],
            (pygame.K_LEFT, pygame.K_SPACE, pygame.K_LSHIFT): ['left', 'A', 'B'],
        }

    def get_action(self, keys_pressed):
        pressed_keys = []
        if keys_pressed[pygame.K_n]:
            pressed_keys = ['NOOP']
        elif keys_pressed[pygame.K_RIGHT] and keys_pressed[pygame.K_SPACE] and keys_pressed[pygame.K_LSHIFT]:
            pressed_keys = ['right', 'A', 'B']
        elif keys_pressed[pygame.K_RIGHT] and keys_pressed[pygame.K_SPACE]:
            pressed_keys = ['right', 'A']
        elif keys_pressed[pygame.K_RIGHT] and keys_pressed[pygame.K_LSHIFT]:
            pressed_keys = ['right', 'B']
        elif keys_pressed[pygame.K_LEFT] and keys_pressed[pygame.K_SPACE] and keys_pressed[pygame.K_LSHIFT]:
            pressed_keys = ['left', 'A', 'B']
        elif keys_pressed[pygame.K_LEFT] and keys_pressed[pygame.K_SPACE]:
            pressed_keys = ['left', 'A']
        elif keys_pressed[pygame.K_LEFT] and keys_pressed[pygame.K_LSHIFT]:
            pressed_keys = ['left', 'B']
        elif keys_pressed[pygame.K_RIGHT]:
            pressed_keys = ['right']
        elif keys_pressed[pygame.K_LEFT]:
            pressed_keys = ['left']
        elif keys_pressed[pygame.K_DOWN]:
            pressed_keys = ['down']
        elif keys_pressed[pygame.K_UP]:
            pressed_keys = ['up']
        elif keys_pressed[pygame.K_SPACE]:
            pressed_keys = ['A']

        if not pressed_keys:
            return 0  # 無有效輸入時返回NOOP

        pressed_keys = sorted(pressed_keys)
        for i, action in enumerate(COMPLEX_MOVEMENT):
            if pressed_keys == sorted(action):
                # print(f"Action: {action}, Index: {i}")
                return i

        return 0  # 無匹配動作時返回NOOP

# 初始化pygame
pygame.init()
screen = pygame.display.set_mode((240, 256))
pygame.display.set_caption("Super Mario Bros")
clock = pygame.time.Clock()

# 初始化環境並應用wrapper
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = SkipAndMax(env, skip=4)
env = Frame_Processing(env)
env = BufferingWrapper(env, n_steps=4)

state = env.reset()  # state.shape = (4, 84, 84)
done = False
input_handler = InputHandler()
print(state.shape)

# 按鍵說明
print("按鍵說明：")
print("N: NOOP（無操作）")
print("右箭頭: 向右")
print("左箭頭: 向左")
print("下箭頭: 下蹲")
print("上箭頭: 向上")
print("空格: A（跳躍）")
print("Shift: B（衝刺）")
print("右+空格: 向右跳")
print("右+Shift: 向右衝刺")
print("右+空格+Shift: 向右跳+衝刺")
print("左+空格: 向左跳")
print("左+Shift: 向左衝刺")
print("左+空格+Shift: 向左跳+衝刺")
print("Ctrl+C: 退出（在pygame窗口按）")
print("注意：請點擊pygame窗口（馬力歐畫面）激活鍵盤焦點！")

# prev_x = None

# 儲存trajectory
trajectory = []
total_reward = 0
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_CTRL:
                print("Ctrl+C detected, exiting...")
                done = True

    keys_pressed = pygame.key.get_pressed()
    action = input_handler.get_action(keys_pressed)

    # 執行環境步驟
    next_state, reward, done, info = env.step(action)
    trajectory.append((state, action, reward, next_state, done))
    total_reward += reward
    # if reward != 0:
    #     print(reward)

    # x_pos = info['x_pos']
    # if prev_x is None:
    #     prev_x = x_pos
    # dx = x_pos - prev_x
    # print(dx)
    # prev_x = x_pos

    # 使用info中的raw_states進行渲染（選擇最後一幀）
    raw_state = info.get('raw_states', [state])[-1]  # 取最後一幀原始state
    state_surface = pygame.surfarray.make_surface(raw_state.swapaxes(0, 1))
    screen.blit(state_surface, (0, 0))
    pygame.display.flip()

    state = next_state
    clock.tick(2)

env.close()
pygame.quit()

print(total_reward)
print(len(trajectory))

play = {
    'trajectory': trajectory,
    'total_reward': total_reward,
}
i = 49
path = f"./human_play/play_{i}.pkl"
with open(path, 'wb') as f:
    pickle.dump(play, f)
print(f"trajectory saved to {path}")
