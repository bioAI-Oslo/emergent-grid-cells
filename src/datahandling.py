import numpy as np
import torch


class CLSampler(torch.utils.data.Sampler):
    """Continual Learning Sampler"""

    def __init__(self, num_environments, num_samples, **kwargs):
        self.num_samples = num_samples
        self.sample_counter = -1
        self.samples_per_environment = num_samples / num_environments

    def __iter__(self):
        def cl_generator():
            for _ in range(len(self)):
                if self.sample_counter == self.num_samples - 1:
                    self.sample_counter = -1
                self.sample_counter += 1
                current_environment_idx = int(
                    self.sample_counter // self.samples_per_environment
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
            while True:
                environment_idx = self.step_counter % self.num_environments
                self.step_counter += 1
                yield environment_idx
        return me_generator()

    def __len__(self):
        return self.num_samples


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, agents, place_cell_ensembles, num_samples, seq_len=20, **kwargs
    ):
        self.agents = agents
        self.place_cell_ensembles = place_cell_ensembles
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        agent = self.agents[index]
        place_cells = self.place_cell_ensembles[index]
        agent.reset()
        for _ in range(self.seq_len):
            agent.step()

        velocities = torch.tensor(agent.velocities[1:], dtype=torch.float32)
        positions = torch.tensor(agent.positions, dtype=torch.float32)
        pc_positions = place_cells.softmax_response(positions)
        init_pc_positions, labels = pc_positions[0], pc_positions[1:]
        return [[velocities, init_pc_positions], labels, positions, index]
