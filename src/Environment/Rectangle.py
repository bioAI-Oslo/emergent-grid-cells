import numpy as np

from Environment import Environment


class Rectangle(Environment):
    def __init__(self, boxsize=(1, 1), res=(32, 32)):
        self.origo = np.array((0, 0))  # bottom left coordinate
        self.boxsize = np.array(boxsize)  # top right coordinate
        self.res = res

        # initialize board
        xx, yy = np.meshgrid(
            np.linspace(self.origo[0], boxsize[0], res[0]),
            np.linspace(self.origo[1], boxsize[0], res[0]),
        )
        self.board = np.stack([xx, yy], axis=-1)

    def uniform_sample(self, ns=1):
        """
        Uniform sampling a 2d-rectangle is trivial with numpy
        """
        return np.random.uniform(self.origo, self.boxsize, size=(ns, 2))

    def near_wall(self, pos, soft_boundary=0.03):
        # Assume agent inside environment.
        d_left_bottom_wall = pos - self.origin
        d_right_top_wall = self.boxsize - pos

        # if sum(abs(escape_direction)) == 0 => not in boundary condition
        #                               == 1 => near one wall
        #                               == 2 => near a corner
        escape_direction = ((d_left_bottom_wall) <= soft_boundary) - (
            (d_right_top_wall) <= soft_boundary
        )
        escape_direction = escape_direction.astype(int) # from bool to int

        scenario = sum(abs(escape_direction))
        if scenario == 2:
            # Find optimal escape direction from an arbitrary 
            # location within a corner 
            box_center = (self.boxsize - self.origo) / 2
            pos_wrt_center = pos - box_center
            optimal_escape_direction = pos_wrt_center - np.sign(pos_wrt_center) * soft_boundary

            return optimal_escape_direction

        # remaining scenarios are already optimal
        return escape_direction

    def inside_environment(self, pos):
        """
        Check if agent is inside the defined environment
        """
        posx,posy=pos
        statement = (self.origo[0] <= posx <= self.boxsize[0]) and (
            self.origo[1] <= posy <= self.boxsize[1]
        )
        return statement

    def walls(self, pos, speed, prev_angle, turn, soft_boundary=0.03):
        """
        Make sure agent is inside environment, and that
        next step (speed, angle) does not take agent
        outside environment.
        """
        # Break program if agent outside box
        # We do not want to train agent in an undefined envinronment.
        # (this should not happen, and thus hopefully this is a redundant test.)
        assert self.inside_environment(posx, posy), "Agent outside box. Pos={}".format(
            posx, posy
        )

        angle = prev_angle + turn
        vel = speed * np.cos(angle), speed * np.sin(angle)
        next_pos = pos + vel

        # clip next_position to within the environment
        next_pos = np.clip(next_pos, self.origo, self.boxsize)

        # escape direction
        ed = self.near_wall(next_pos, soft_boundary)

        alternative_vel = speed * np.cos(-angle), speed * np.sin(-angle)

        score1 = ed @ np.array(vel)
        score2 = ed @ np.array(alternative_vel)

        if score2 > score1:
            turn = - turn

        if sum(abs(ed)) > 0:
            speed = speed * 0.25

        return speed, turn


if __name__ == "__main__":
    print(Rectangle())
