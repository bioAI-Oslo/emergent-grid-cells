import numpy as np
import torch


class CLSampler(torch.utils.data.Sampler):
    """Continual Learning Sampler"""

    def __init__(self, num_environments, num_samples, num_epochs, **kwargs):
        self.num_samples = num_samples
        self.num_environments = num_environments
        self.num_epochs = num_epochs
        self.epoch_counter = -1
        self.epochs_per_environment = num_epochs / num_environments

    def __iter__(self):
        def cl_generator():
            self.epoch_counter += 1
            for sample_i in range(len(self)):
                current_environment_idx = int(
                    self.epoch_counter // self.epochs_per_environment
                )
                yield current_environment_idx
        return cl_generator()

    def __len__(self):
        return self.num_samples


class MESampler(torch.utils.data.Sampler[int]):
    """Multi Environment Sampler"""

    def __init__(self, num_environments, num_samples, **kwargs):
        self.num_environments = num_environments
        self.num_samples = num_samples
        self.step_counter = 0

    def __iter__(self):
        def me_generator():
            for _ in range(len(self)):
                environment_idx = self.step_counter % self.num_environments
                self.step_counter += 1
                yield environment_idx
        return me_generator()

    def __len__(self):
        return self.num_samples


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, agents, pc_ensembles, num_samples, seq_len=20, **kwargs
    ):
        self.agents = agents
        self.pc_ensembles = pc_ensembles
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        agent = self.agents[index]
        place_cells = self.pc_ensembles[index]
        agent.reset()
        for _ in range(self.seq_len):
            agent.step()

        velocities = torch.tensor(agent.velocities[1:], dtype=torch.float32)
        positions = torch.tensor(agent.positions, dtype=torch.float32)
        pc_positions = place_cells.softmax_response(positions)
        init_pc_positions, labels = pc_positions[0], pc_positions[1:]
        return [[velocities, init_pc_positions], labels, positions, index]


if __name__ == '__main__':
    num_epochs = 5
    clsampler = CLSampler(num_environments=3, num_samples = 10, num_epochs = num_epochs)
    for epoch in range(num_epochs):
        for env_i in clsampler:
            print(env_i)

