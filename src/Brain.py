import torch

class Brain:
    def __init__(self, env, npcs, sigma=0.12, DoG=False, surround_scale=2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary", dtype=torch.float32):
        """
        Args:
                env: ABCEnvironment class instance
                npcs: Number of place cells
                sigma: Array-like or scalar. Tuning curve of place cells
        """
        self.env = env
        self.pcs = env.sample_uniform(npcs)  # sample place cell
        self.pcs = torch.tensor(self.pcs, dtype=dtype)
        self.npcs = npcs
        self.sigma = sigma
        self.DoG = DoG
        self.surround_scale = surround_scale
        self.p = p
        self.compute_mode = compute_mode

    def softmax_response(self, pos):
        """Place cell response as modelled by Sorscher"""
        if len(pos.shape) == 1:
            pos = pos[None]
            out_shape = self.npcs
        elif len(pos.shape) == 2:
            out_shape = [pos.shape[0], self.npcs]
        elif len(pos.shape) == 3:
            out_shape = list(pos.shape[:-1]) + [self.npcs]
            pos = pos.reshape(pos.shape[0] * pos.shape[1], pos.shape[-1])

        # distance to place cell center
        # dists = torch.cdist(pos, self.pcs, p=self.p, compute_mode=self.compute_mode)
        dists = torch.sum((pos[:,None] - self.pcs[None])**2, axis=-1)
        activity = torch.nn.functional.softmax(-dists / (2 * self.sigma ** 2), dim=-1)

        if self.DoG:
            activity -= torch.nn.functional.softmax(-dists / (2 * self.surround_scale * self.sigma ** 2), dim=-1)

            # after DoG, activity is not a probability dist anymore
            # shift and rescale s.t it becomes a prob dist again.
            activity -= torch.min(activity, dim=-1, keepdim=True).values # returns idxs and values
            activity /= torch.sum(activity, dim=-1, keepdim=True)

        return activity.reshape(out_shape)

    def to_euclid(self, activity, k=3):
        """
        Decode place-cell activity to Euclidean coordinates - following Sorscher.
        OBS! This is an approximation to the actual Euclidean location,
        by considering the top k place-cell activities as if the agent is located
        at the average k place-cell center location
        """
        _, idxs = torch.topk(activity, k, dim=-1)
        return torch.mean(self.pcs[idxs], axis=-2)
        
