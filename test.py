from env_wrapper import make_env
from train import Agent, Transition

env = make_env()

agent = Agent(env.observation_space.shape, env.action_space.n)

agent.load_model("./models/rainbow_icm.pth", eval_mode=True)

state = env.reset()
done = False
total_reward = 0
while not done:
    state, reward, done, info = env.step(agent.act(state))
    # print(info)
    env.render()
    total_reward += reward

env.close()

print(total_reward)
