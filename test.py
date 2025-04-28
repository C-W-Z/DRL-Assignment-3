from env_wrapper import make_env
from train import Agent, Transition
import pygame

env = make_env()

agent = Agent(env.observation_space.shape, env.action_space.n)
agent.load_model("./models/rainbow_icm.pth", eval_mode=True)

pygame.init()
screen = pygame.display.set_mode((240, 256))
pygame.display.set_caption("Super Mario Bros")
clock = pygame.time.Clock()

state = env.reset()
done = False
total_reward = 0
while not done:
    state, reward, done, info = env.step(agent.act(state))
    # print(info)
    # env.render()
    total_reward += reward

    state_surface = pygame.surfarray.make_surface(state.swapaxes(0, 1))
    screen.blit(state_surface, (0, 0))
    pygame.display.flip()

    clock.tick(60)

env.close()

print(total_reward)
