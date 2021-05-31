import collections
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.spatial.distance import euclidean


class Brain:
    def __init__(self, env, npcs, sigma):
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

    def d_pcc(self, pos, pc, wall_exceptions=[]):
        """
        (Shortest geodesic?) distance to a place cell center
        """
        return euclidean(pos,pc) # skip-geodesic

        # GEODESIC: NOT COMPLETE / UNDER DEV
        ni, wall = self.env.crash_point(pos, pc - pos, wall_exceptions)
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
            d1 = self.d_pcc(pos, bias_branch, wall_exceptions=[]) + self.d_pcc(
                bias_branch, pc, wall_exceptions=[]
            )

        if end_corner:
        	d2 = np.inf
        else:
            d2 = self.d_pcc(pos, end_branch, wall_exceptions=[]) + self.d_pcc(
                end_branch, pc, wall_exceptions=[]
            )

        if d1 == np.inf and d2 == np.inf:
        	raise NotImplementedError('Shortest distance in env with dead ends not implemented.')

        return min(d1, d2)

    def multivar_norm_response(self, pos, pc):
        """
        Returns activity where place cell tuning curves are
        modelled as a multivariate gaussian func
        """
        activity = multivariate_normal.pdf(pos, pc, np.diag(self.sigma))
        activity = euclidean(activity, np.zeros_like(activity))  # magnitude
        return activity

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
