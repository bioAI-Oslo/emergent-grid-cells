import numpy as np
from ctimeit import ctimeit  # custom timing of functions


class Agent:
    def __init__(self, angle0, speed0, p0=None):
        self.speeds = np.array([speed0])  # speed history
        self.hds = np.array([angle0])  # head direction history
        self.turns = np.zeros(0)  # turn direction history
        self._velocities = np.zeros((0, 2))  # velocity history
        self._positions = np.zeros((1, 2)) if p0 is None else p0  # position history

        # add size to rat? e.g as an ellipsoid?

    def step(self, walls, record_step=True):
        """
        Sample a velocity vector - indirectly through speed
        and angle, i.e. (s,phi). The angle is an offset to
        the angle at the previous time step.
        """
        # constants used by Sorscher:
        dt = 0.02
        sigma = 5.76 * 2  # stdev rotation velocity (rads/sec)
        b = 0.13 * 2 * np.pi  # forward velocity rayleigh dist scale (m/sec)
        mu = 0  # turn angle bias

        new_speed = np.random.rayleigh(b) * dt
        new_turn = np.random.normal(mu, sigma) * dt

        new_speed, new_turn, next_pos = walls(
            self.positions[-1], self.hds[-1], new_speed, new_turn
        )

        new_hd = np.mod(self.hds[-1] + new_turn, 2 * np.pi)

        if record_step:
            self.speeds = np.append(self.speeds, new_speed)
            self.hds = np.append(self.hds, new_hd)
            self.turns = np.append(self.turns, new_turn)
            #print(self._positions.shape,next_pos)
            self._positions = np.append(self._positions, next_pos[None], axis=0)
            #self._positions = np.concatenate([self._positions, next_pos])
            #print(self._positions.shape,next_pos)

        return new_speed, new_hd

    @property
    def velocities(self):
        """
        Euclidean velocity history
        """
        idx0 = self._velocities.shape[0]
        if idx0 < self.speeds.shape[0]:
            return self._velocities

        euclidean_direction = np.stack([np.cos(self.hds[idx0:]), np.sin(self.hds[idx0:])], axis=-1)
        velocity = euclidean_direction * self.speeds[idx0:][..., None]
        self._velocities = np.concatenate([self._velocities, velocity])
        return self._velocities

    @property
    def positions(self):
        """
        Path integration (Euclidean position) history
        """
        idx0 = self._positions.shape[0]
        if idx0 < self.speeds.shape[0]:
            return self._positions

        #print(self.velocities[-1])
        delta_p = np.cumsum(self.velocities[idx0:], axis=0)
        self._positions = np.concatenate(
            [self._positions, delta_p + self._positions[-1]]
        )
        return self._positions


if __name__ == "__main__":
    """
    Simple tests/inspection of class methods
    """
    ag = Agent(0, 0)

    for i in range(100000):
        ag.step()

    print(ag.speeds)
    print(ag.hds)
    print(ag.velocities)
    print(ag.positions.shape)
