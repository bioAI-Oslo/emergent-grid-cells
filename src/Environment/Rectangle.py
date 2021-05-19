import numpy as np
from scipy.spatial.distance import euclidean

from Environment import Environment


class Rectangle(Environment):
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

    def crash_point(self, pos, vel):
        """
        Treat current position and velocity, and walls as line-segments.
        We can then find where the agent will crash on each wall
        by solving a system of linear equations for each wall.
        """
        for wall_key in self.walls:
            matrix = np.array([vel, self.walls[wall_key]["slope"]])
            vector = pos - self.walls[wall_key]["bias"]
            try:
                solution = np.linalg.solve(matrix, vector)
            except np.linalg.LinAlgError as e:
                # Singular matrix (Agent trajectory parallell to wall)
                print(e)
                continue

            solution = self.walls[wall_key]["bias"] + solution[1]*self.walls[wall_key]["slope"]

            # Check if solution is inside environment,
            # then choose the wall that is along the trajectory path
            # (as opposed to the wall in the opposite direction)
            #print("pos,vel,sol,wall,score", pos, vel, solution,wall_key, (vel @ (solution - pos)))
            if self.inside_environment(solution) and (vel @ (solution - pos)) >= 0:
                #print("ok")
                return solution

    def inside_environment(self, pos):
        """
        Check if agent is inside the defined environment
        """
        posx, posy = pos
        statement = (self.origo[0] <= posx <= self.boxsize[0]) and (
            self.origo[1] <= posy <= self.boxsize[1]
        )
        return statement

    def near_wall(self, pos):
        """
        Returns whether agent is near a wall, a corner or neither. Also
        returns optimal direction for escaping a wall or a corner
        relative to pos.
        """
        # Assume agent inside environment.
        d_left_bottom_wall = pos - self.origo
        d_right_top_wall = self.boxsize - pos

        # if sum(abs(escape_direction)) == 0 => not in boundary condition
        #                               == 1 => near one wall
        #                               == 2 => near a corner
        escape_direction = ((d_left_bottom_wall) <= self.soft_boundary).astype(int) - (
            (d_right_top_wall) <= self.soft_boundary
        ).astype(int)
        escape_direction = escape_direction.astype(int)  # from bool to int

        scenario = sum(abs(escape_direction))
        if scenario == 2:
            # Find optimal escape direction from an arbitrary
            # location within an arbitrary corner
            box_center = (self.boxsize - self.origo) / 2
            pos_wrt_center = pos - box_center
            # optimal_escape_direction = pos_wrt_center - np.sign(pos_wrt_center) * (
            #    box_center - self.soft_boundary
            # )  # sorcery
            optimal_escape_direction = -np.sign(pos_wrt_center) * (
                box_center - self.soft_boundary
            )  # sorcery

            return scenario, optimal_escape_direction

        # remaining scenarios are already optimal
        return scenario, escape_direction

    def avoid_walls(self, pos, hd, speed, turn):
        # escape direction
        scenario, ed = self.near_wall(pos)

        if scenario > 0:
            # determine which turn direction is most similar
            # to the optimal escape direction
            score_p = ed @ [np.cos(hd + turn), np.sin(hd + turn)]
            score_n = ed @ [np.cos(hd - turn), np.sin(hd - turn)]
            if score_n > score_p:
                turn = -turn  # turn towards optimal escape direction

            # slow down when near walls
            #speed = speed * 0.25  # as used by Sorscher

        direction = np.array([np.cos(hd + turn), np.sin(hd + turn)])
        crash_point = self.crash_point(pos, direction)

        
        # speed is maximum half the distance to the crash point
        speed = min(speed, euclidean(pos, crash_point) / 2)

        next_pos = pos + direction * speed
        #print("pos, crash_point, next_pos, speed, direction", pos, crash_point,next_pos,speed,direction)

        return speed, turn, next_pos

if __name__ == "__main__":
    print(Rectangle())
