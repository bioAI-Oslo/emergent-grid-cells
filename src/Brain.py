import numpy as np
import tensorflow as tf

from scipy.stats import norm, multivariate_normal
from scipy.spatial.distance import euclidean, cdist


class Brain:
    def __init__(self, env, npcs, sigma=0.12):
        """
        Args:
                env: ABCEnvironment class instance
                npcs: Number of place cells
                sigma: Array-like or scalar. Tuning curve of place cells
        """
        self.env = env
        self.pcs = env.sample_uniform(npcs)  # sample place cell
        self.npcs = npcs
        self.sigma = sigma

    def __call__(
        self, pos, lateral_inhibition=True, activity_model=None, metric="euclidean"
    ):
        """Brain response/basis: dist U tuning-curve"""
        # cdist() takes matrices (2D) inputs. reshape to satisfy
        if len(pos.shape) == 1:
            pos = pos[None]
            out_shape = self.npcs
        elif len(pos.shape) == 2:
            out_shape = [pos.shape[0], self.npcs]
        elif len(pos.shape) == 3:
            out_shape = list(pos.shape[:-1]) + [self.npcs]
            pos = pos.reshape(np.prod(pos.shape[:-1]), pos.shape[-1])

        # distance to place cell center
        dists = cdist(pos, self.pcs, metric=metric)

        # unit place-cell activity (wrt. tuning curve)
        activity = (
            self.ricker_response(dists)
            if activity_model is None
            else activity_model(dists)
        )

        # Bio: Lateral inhibition. Engineering: Place-cell activity ensemble
        # must satisfy probability distribution requirements (else cross-entropy
        # is invalid).
        if lateral_inhibition:
            activity -= np.min(activity, axis=-1, keepdims=True)
            activity /= np.sum(activity, axis=-1, keepdims=True)

        return activity.reshape(out_shape)

    def softmax_response(self, pos, DoG=False, surround_scale=2, metric="euclidean"):
        """Place cell response as modelled by Sorscher"""
        if len(pos.shape) == 1:
            pos = pos[None]
            out_shape = self.npcs
        elif len(pos.shape) == 2:
            out_shape = [pos.shape[0], self.npcs]
        elif len(pos.shape) == 3:
            out_shape = list(pos.shape[:-1]) + [self.npcs]
            pos = pos.reshape(np.prod(pos.shape[:-1]), pos.shape[-1])

        # distance to place cell center
        dists = cdist(pos, self.pcs, metric=metric)

        # cast to tf.Tensor, dtype=tf.float32, use tf's softmax func
        # and recast to numpy array
        activity = tf.keras.activations.softmax(
            tf.convert_to_tensor(-dists / (2 * self.sigma ** 2), dtype=tf.float32)
        ).numpy()

        if DoG:
            activity -= tf.keras.activations.softmax(
                tf.convert_to_tensor(
                    -dists / (2 * surround_scale * self.sigma ** 2), dtype=tf.float32
                )
            ).numpy()

            # after DoG, activity is not a probability dist anymore
            # shift and rescale s.t it becomes a prob dist again.
            activity -= np.min(activity, axis=-1, keepdims=True)
            activity /= np.sum(activity, axis=-1, keepdims=True)

        return activity.reshape(out_shape)

    def to_euclid(self, activity, k=3):
        """
        Decode place-cell activity to Euclidean coordinates - following Sorscher.
        OBS! This is an approximation to the actual Euclidean location,
        by considering the top k place-cell activities as if the agent is located
        at the average k place-cell center location
        """
        mus = tf.math.top_k(activity, k=k).indices.numpy()
        return np.mean(self.pcs[mus], axis=-2)

    def inverse(self):
        """To Euclidean coordinates from place-cell coordinates"""
        pass  # not implemented yet

    def d_pcc(self, pos, pc):
        """
        (Shortest geodesic?) distance to a place cell center
        """
        return euclidean(pos, pc)  # skip-geodesic

        # GEODESIC: NOT COMPLETE / UNDER DEV
        ni, wall = self.env.crash_point(pos, pc - pos)
        atcf = euclidean(pos, pc)
        if atcf <= euclidean(pos, ni) or euclidean(pos, ni) < 1e-12:
            # no obstruction => as the crow flies (atcf)
            # either from an open position in the env or
            # from sitting on a wall (< 1e-12. i.e. +- epsilon).
            return atcf

        # else
        bias_branch, end_branch = wall.bias, wall.end
        bias_corner, end_corner = wall.iscorner

        # OBS! Corners are hard... Thus only walls with-non corners or free walls
        # with maximum one corner (and no dead-ends) have geodesic distance
        # currently implemented OBS!
        if bias_corner:
            d1 = np.inf
        else:
            d1 = self.d_pcc(pos, bias_branch) + self.d_pcc(bias_branch, pc)

        if end_corner:
            d2 = np.inf
        else:
            d2 = self.d_pcc(pos, end_branch) + self.d_pcc(end_branch, pc)

        if d1 == np.inf and d2 == np.inf:
            raise NotImplementedError(
                "Shortest distance in env with dead ends not implemented."
            )

        return min(d1, d2)

    def multivar_norm_response(self, pos, pcs=None):
        """
        Returns activity where place cell tuning curves are
        modelled as a multivariate gaussian func
        """
        pcs = self.pcs if pcs is None else pcs
        activity = multivariate_normal.pdf(pos, pcs, np.diag(self.sigma))
        activity = euclidean(activity, np.zeros_like(activity))  # magnitude
        # return activity
        pass  # Currently incomplete func.

    def norm_response(self, d):
        """
        Returns activity where place cell tuning curves are
        modelled as a gaussian func
        """
        activity = norm.pdf(d, 0, self.sigma)
        return activity

    def ricker_response(self, d):
        """
        Returns activity where place cell tuning curves are
        modelled as a ricker (DoG/LoG) func (zero-mean/sum)

        (There also exists 2d-ricker if necessary)
        """
        activity = 2 / (np.sqrt(3 * self.sigma) * np.pi ** (1 / 4))
        activity *= 1 - (d / self.sigma) ** 2
        activity *= np.exp(-(d ** 2) / (2 * self.sigma ** 2))
        return activity
