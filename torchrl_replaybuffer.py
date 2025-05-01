from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
import torch

class ReplayBuffer:
    def __init__(self, obs_shape, size, batch_size, n_step, gamma, device=None):
        self.obs_shape = obs_shape
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(size)
        )

    @property
    def size(self):
        return len(self.buffer)

    def store(self, obs, action, reward, next_obs, done):
        data = TensorDict({
            'obs':       torch.tensor(obs, dtype=torch.float32),
            'acts':      torch.tensor([action], dtype=torch.int64),
            'rews':      torch.tensor([reward], dtype=torch.float32),
            'next_obs':  torch.tensor(next_obs, dtype=torch.float32),
            'done':      torch.tensor([done], dtype=torch.bool),
        }, batch_size=[]).to(self.device)

        self.buffer.add(data)

    def sample_batch(self):
        batch = self.buffer.sample(self.batch_size)

        return {
            'obs':       batch['obs'],
            'acts':      batch['acts'].squeeze(1),
            'rews':      batch['rews'].squeeze(1),
            'next_obs':  batch['next_obs'],
            'done':      batch['done'].float().squeeze(1),
        }

    def get_state(self):
        return {
            'storage': self.buffer._storage._storage,
        }

    def set_state(self, state):
        self.buffer._storage._storage = state['storage']
