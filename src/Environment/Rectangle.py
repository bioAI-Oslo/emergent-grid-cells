import numpy as np
from scipy.spatial.distance import euclidean

from .ABCEnvironment import ABCEnvironment
from .methods import intersect, projection_rejection


class Rectangle(ABCEnvironment):
    def __init__(self, boxsize=(1, 1), soft_boundary=0.3):
        self.origo = np.array((0, 0))  # bottom left coordinate
        self.boxsize = np.array(boxsize)  # top right coordinate
        self.soft_boundary = soft_boundary

        # init walls
        self.walls = {
            "w1": {
                "bias": self.origo,
                "slope": np.array([self.origo[0], self.boxsize[1]]),
            },
            "w2": {
                "bias": np.array([self.origo[0], self.boxsize[1]]),
                "slope": np.array([self.boxsize[0], self.origo[1]]),
            },
            "w3": {
                "bias": np.array([self.boxsize[0], self.origo[1]]),
                "slope": np.array([self.origo[0], self.boxsize[1]]),
            },
            "w4": {
                "bias": self.origo,
                "slope": np.array([self.boxsize[0], self.origo[1]]),
            },
        }

    def get_board(self, res=(32, 32)):
        # initialize board
        xx, yy = np.meshgrid(
            np.linspace(self.origo[0], boxsize[0], res[0]),
            np.linspace(self.origo[1], boxsize[1], res[1]),
        )
        return np.stack([xx, yy], axis=-1)

    def sample_uniform(self, ns=1):
        """
        Uniform sampling a 2d-rectangle is trivial with numpy
        """
        return np.random.uniform(self.origo, self.boxsize, size=(ns, 2))

    def add_wall(self, wall_key, wall):
        """Add wall to walls."""
        self.walls[wall_key] = wall

    def inside_environment(self, pos):
        """
        Check if agent is inside the defined environment
        """
        return (self.boxsize >= (pos - self.origo)).all()

    def crash_point(self, pos, vel):
        """
        Treat current position and velocity, and walls as line-segments.
        We can then find where the agent will crash on each wall
        by solving a system of linear equations for each wall.
        """
        nearest_intersection = None
        for wall in self.walls.values():
            intersection, valid_intersect = intersect(
                pos, vel, *wall.values(), [0, np.inf], [0, 1]
            )

            if valid_intersect:  # along animal trajectory and inside env
                if (nearest_intersection is None) or (
                    euclidean(pos, nearest_intersection) > euclidean(pos, intersection)
                ):
                    nearest_intersection = intersection

        return nearest_intersection

    def wall_rejection(self, pos):
        """
        Walls reject agent when it comes too close.
        ed (escape direction) is the direction the agent is rejected towards.
        """
        ed = np.zeros(2)
        for wallname, wall in self.walls.items():
            proj, rej = projection_rejection(pos - wall["bias"], wall["slope"])

            # could neglect this if-test, but it saves a few unnecessary compuations
            if "free_wall" in wallname:

                # Projection-vector must have correct direction and be contained
                # in the line-segment
                direction = proj @ wall["slope"]
                if not (
                    (direction >= 0) and (direction <= wall["slope"] @ wall["slope"])
                ):
                    continue

            d = euclidean(self.origo, rej)
            ed += int(d <= self.soft_boundary) * (rej / d)  # unit-rejection

        return ed

    def avoid_walls(self, pos, hd, speed, turn):
        # --- Regulate turn ---
        ed = self.wall_rejection(pos)

        # score next animal direction wrt. wall escape direction
        score_p = ed @ [np.cos(hd + turn), np.sin(hd + turn)]
        score_n = ed @ [np.cos(hd - turn), np.sin(hd - turn)]
        if score_n > score_p:
            turn = -turn  # turn away from wall

        # --- Regulate speed ---
        direction = np.array([np.cos(hd + turn), np.sin(hd + turn)])
        intersection = self.crash_point(pos, direction)

        # speed is maximum half the distance to the crash point
        speed = min(speed, euclidean(pos, intersection) / 2)
        return speed, turn
