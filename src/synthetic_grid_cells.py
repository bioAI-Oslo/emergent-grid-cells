from typing import Callable
import numpy as np
import numpy.ma as ma
import numpy.linalg as npl
import matplotlib.pyplot as plt


def rotation_matrix(theta, degrees=True, **kwargs) -> np.ndarray:
    """
    Creates a 2D rotation matrix for theta
    Parameters
    ----------
    theta : float
        angle offset wrt. the cardinal x-axis
    degrees : boolean
        Whether to use degrees or radians
    Returns
    -------
    rotmat : np.ndarray
        the 2x2 rotation matrix
    Examples
    --------
    >>> import numpy as np
    >>> x = np.ones(2) / np.sqrt(2)
    >>> rotmat = rotation_matrix(45)
    >>> tmp = rotmat @ x
    >>> eps = 1e-8
    >>> np.sum(np.abs(tmp - np.array([0., 1.]))) < eps
    True
    """
    # convert to radians
    theta = theta * np.pi / 180 if degrees else theta
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def grid_cell(
    phase_offset,
    orientation_offset=0,
    f=1,
    rot_theta=60,
    n_comps=3,
    non_negative=True,
    add=True,
    **kwargs
) -> Callable:
    """
    Grid cell pattern constructed from three interacting 2D (plane) vectors
    with 60 degrees relative orientational offsets.
    See e.g. the paper: "From grid cells to place cells: A mathematical model"
    - Moser & Einevoll 2005
    Parameters
    ----------
    phase_offset : np.ndarray
        2D-array. Spatial (vector) phase offset of the grid pattern. Note that
        the grid pattern is np.array([f,f])-periodic, so phase-offsets are also
        f-periodic
    orientation_offset : float
        First plane vector is default along the cardinal x-axis. Phase-offset
        turns this plane vector counter clockwise (default in degrees, but
        can use **kwargs - degrees=False to use radians)
    f : float
        Spatial frequency / periodicity. f=1 makes the grid cell unit-periodic
    Returns
    -------
    grid_cell_fn : function
        A grid cell function which can be evaluated at locations r
    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros(2)
    >>> gc = grid_cell()
    >>> gc(x)
    3.0
    """
    relative_R = rotation_matrix(rot_theta)
    #
    # Grid Cell lattice is the Reciprocal Lattice of the 
    # Generating Lattice spanned by the k-Vectors.
    # This Generating Lattice has Spacial Period 2 / f along its Basis Vectors (e.g. k0 and k1)
    # The Reciprocal Lattice of the hexagonal Lattice is the Lattice itself, offset by
    # 30.0 degrees in Orientation.
    # The Period of this Reciprocal Lattice (Grid Cell Pattern Lattice)
    # is the Radius of the Wigner-Seitz Primitive Cell.
    # The 
    # 

    # create Grid Cell lattice from Generting Lattice
    init_R = rotation_matrix(orientation_offset + 30.0, **kwargs)

    # define Basis and Wave-Vectors of generating Lattice
    k1 = np.array([1.0, 0.0])  # init wave vector. unit length in x-direction
    k1 = init_R @ k1
    ks = np.array([npl.matrix_power(relative_R, k) @ k1 for k in range(n_comps)])
    ks *= 2 * np.pi  # spatial angular frequency (unit-movement in space is one period)
    
    # user-defined spatial frequency
    # relate Grid Cell Pattern frequency to frequency of the
    # generating lattice
    f_gen = f / np.cos(np.pi / 6)
    ks *= f_gen  

    def grid_cell_fn(r):
        """
        Grid cell function with fixed parameters given by outer function. I.e.
        a mapping from some spatial coordinates r to grid cell activity.
        Parameters
        ----------
        r : np.ndarray
            [1,2 or 3]D array. For a 3D array, the shape is typically (Ng,Ng,2).
            A tensor of 2D-spatial coordinates.
        Returns:
        grid_cell_activity: np.ndarray or float
            [0,1 or 2]D array. For a 2D array, the shape is typically (Ng,Ng).
            The activity of this grid cell across all spatial coordinates in
            the grid (Ng,Ng).
        """
        r0 = phase_offset
        if r0.ndim == 2 and r.ndim > 2:
            for i in range(1, r.ndim):
                r0 = r0[:, None]
        if not add:
            return np.cos((r - r0) @ ks.T)

        activity = np.sum(np.cos((r - r0) @ ks.T), axis=-1)
        if non_negative:
            activity = np.maximum(activity, 0)
        else:
            # scale to [0,1]
            activity = 2 * (activity / 3 + 0.5) / 3
        return activity

    return grid_cell_fn


class GridModule:
    def __init__(
        self, center, orientation_offset=0, f=1, non_negative=True, add=True, **kwargs
    ):
        self.center = center
        self.orientation_offset = orientation_offset
        self.f = f # frequency of the actual Grid Cell Pattern
        self.non_negative = non_negative
        self.add = add

        # define module outer hexagon
        self.outer_radius = 1 / self.f
        self.outer_hexagon = Hexagon(self.outer_radius, orientation_offset, center)
        # define module inner hexagon based on minimum enclosing circle of Wigner-Seitz cell
        # of a hexagonal lattice with 30 degrees orientation offset to the outer hexagon
        self.inner_radius = 1 / (2 * self.f * np.cos(np.pi / 6))
        self.inner_hexagon = Hexagon(self.inner_radius, orientation_offset + 30, center)

    def init_module(self, phase_offsets):
        self.phase_offsets = phase_offsets
        self.grid_cell_fn = grid_cell(
            phase_offset=phase_offsets,
            orientation_offset=self.orientation_offset,
            f=self.f,
            rot_theta=60,
            n_comps=3,
            non_negative=self.non_negative,
            add=self.add,
        )

    def __call__(self, r):
        return self.grid_cell_fn(r)

    def plot(self, fig, ax):
        self.inner_hexagon.plot(fig, ax)
        self.outer_hexagon.plot(fig, ax)


class Hexagon:
    def __init__(self, radius, orientation_offset, center):
        self.radius = radius
        self.orientation_offset = orientation_offset
        self.center = center
        self.area = 3 * np.sqrt(3) * radius * radius / 2

        # create hexagonal points
        rotmat60 = rotation_matrix(60, degrees=True)
        rotmat_offset = rotation_matrix(orientation_offset, degrees=True)
        hpoints = np.array([radius, 0])  # start vector along cardinal x-axis
        hpoints = rotmat_offset @ hpoints
        hpoints = [hpoints]
        for _ in range(5):
            hpoints.append(rotmat60 @ hpoints[-1])
        self.hpoints = np.array(hpoints)

    def is_in_hexagon(self, point):
        """
        Check if a 2d-point is within a hexagon defined by its 6
        points 'hpoints' with phase 'center'.
        """
        u2, v2 = self.center, point - self.center
        hpoints = self.hpoints + self.center
        for i in range(6):
            # loop each hexagonal side/edge
            u1 = hpoints[i]
            v1 = hpoints[(i + 1) % 6] - u1
            _, intersect_inside_hexagon = intersect(
                u1, v1, u2, v2, constraint1=[0, 1], constraint2=[0, 1]
            )
            if intersect_inside_hexagon:
                return False
        return True

    def sample(self, N):
        # sample points within hexagon
        samples = np.zeros((N, 2))
        for i in range(N):
            sample_square = np.random.uniform(-self.radius, self.radius, 2)
            while not self.is_in_hexagon(sample_square):
                sample_square = np.random.uniform(-self.radius, self.radius, 2)
            samples[i] = sample_square
        return samples

    def plot(self, fig, ax, color="blue"):
        hpoints = self.hpoints + self.center
        for i in range(len(hpoints)):
            ax.plot(*hpoints[i : (i + 2)].T, color=color)
        last_line = np.array([hpoints[-1], hpoints[0]])
        ax.plot(*last_line.T, color=color)

        ax.set_aspect("equal")
        return fig, ax


def intersect(
    u1, v1, u2, v2, constraint1=[-np.inf, np.inf], constraint2=[-np.inf, np.inf]
):
    """
    Calculate intersection of two line segments defined as:
    l1 = {u1 + t1*v1 : u1,v1 in R^n, t1 in constraint1 subseq R},
    l2 = {u2 + t2*v2 : u2,v2 in R^n, t2 in constraint2 subseq R}
    Args:
        u1: bias of first line-segment
        v1: "slope" of first line-segment
        u2: bias of second line-segment
        v2: "slope" of first line-segment
        constraint1: 2d-array(like) of boundary points
                     for the "t-values" of the first line-segment
        constraint1: 2d-array(like) of boundary points
                     for the "t-values" of the second line-segment
    """
    matrix = np.array([v1, -v2]).T
    vector = u2 - u1
    try:
        solution = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError as e:
        # Singular matrix (parallell line segments)
        print(e)
        return None, False

    # check if solution satisfies constraints
    if (constraint1[0] <= solution[0] <= constraint1[1]) and (
        constraint2[0] <= solution[1] <= constraint2[1]
    ):
        return u1 + solution[0] * v1, True

    return u1 + solution[0] * v1, False
