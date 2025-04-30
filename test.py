from env_wrapper import make_env
from train import Agent

env = make_env(life_episode=True)

agent = Agent(env.observation_space.shape, env.action_space.n)
agent.load_model("./models/d3qn_per_bolzman.pth", eval_mode=True)

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