import pickle

id = 33
path = f"./human_play/play_{id}.pkl"
with open(path, 'rb') as f:
    play = pickle.load(f)
    print(play['total_reward'])
trajectory = play['trajectory']
actions = []
for t in trajectory:
    actions.append(t[1])

s = {
    'actions': actions,
    'total_reward': play['total_reward']
}

path = f"./human_play/actions_{id}.pkl"
with open(path, 'wb') as f:
    pickle.dump(s, f)
print(f"actions saved to {path}")
