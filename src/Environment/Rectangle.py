import numpy as np
from scipy.spatial.distance import euclidean

from Environment import Environment


class Rectangle(Environment):
    def __init__(self, boxsize=(1, 1), soft_boundary=0.3):
        self.origo = np.array((0, 0))  # bottom left coordinate
        self.boxsize = np.array(boxsize)  # top right coordinate
        self.soft_boundary = soft_boundary

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
            #optimal_escape_direction = pos_wrt_center - np.sign(pos_wrt_center) * (
            #    box_center - self.soft_boundary
            #)  # sorcery
            optimal_escape_direction = - np.sign(pos_wrt_center) * (
                box_center - self.soft_boundary
            )  # sorcery

            return scenario, optimal_escape_direction

        # remaining scenarios are already optimal
        return scenario, escape_direction

    def walls(self, pos, hd, speed, turn):
        # escape direction
        scenario, ed = self.near_wall(pos)

        next_hd = hd + turn
        vel = speed * np.cos(next_hd), speed * np.sin(next_hd)

        #print(scenario, ed)
        if scenario > 0:

            speed = speed * 0.25  # as used by Sorscher
            # Want to change speed to max half distance to nearest
            # wall, however.

            # determine which turn direction is most similar
            # to the optimal escape direction
            next_hd_n = hd - turn
            vel_n = speed * np.cos(next_hd_n), speed * np.sin(next_hd_n)
            score_p = ed @ np.array(vel)
            score_n = ed @ np.array(vel_n)
            if score_n > score_p:
                turn = -turn  # turn towards optimal escape direction
                next_hd = next_hd_n
                vel = vel_n

        # clip next_position to within the environment
        next_pos = np.clip(pos + vel, self.origo, self.boxsize)
        #next_pos -= np.sign(next_pos) * 1e-06
        speed = euclidean(next_pos, pos)

        print(pos, next_pos, speed)
        # print("ST:", speed, turn)
        # print(speed * np.cos(hd + turn), speed * np.sin(hd + turn))

        return speed, turn

    def walls_old(self, pos, prev_angle, speed, turn):
        """
        Make sure agent is inside environment, and that
        next step (speed, angle) does not take agent
        outside environment.
        """
        # Check if agent is inside the defined environment
        assert self.inside_environment(pos), "Agent outside box. Position={}".format(
            pos
        )

        #
        angle = prev_angle + turn
        vel = speed * np.cos(angle), speed * np.sin(angle)
        next_pos = pos + vel

        # clip next_position to within the environment
        next_pos = np.clip(next_pos, self.origo, self.boxsize)

        # escape direction
        ed = self.near_wall(next_pos, self.soft_boundary)

        #
        alternative_vel = speed * np.cos(-angle), speed * np.sin(-angle)
        score1 = ed @ np.array(vel)
        score2 = ed @ np.array(alternative_vel)
        if score2 > score1:
            turn = -turn

        if sum(abs(ed)) > 0:
            speed = speed * 0.25

        return speed, turn


if __name__ == "__main__":
    print(Rectangle())
