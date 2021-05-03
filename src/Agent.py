import numpy as np
from ctimeit import ctimeit  # custom timing of functions


class Agent:
    def __init__(self, angle0, speed0, p0=None):
        self._speed_history = [speed0]  # ignored atm..
        self._angle_history = [angle0]
        self._velocity_history = np.zeros((0, 2))  # empty 2d-array
        self._position_history = self.p0 = np.zeros((1, 2)) if p0 is None else p0

        # add size to rat? e.g as an ellipsoid?

    def step(self, record_step=True):
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
        new_angle = self._angle_history[-1] + np.random.normal(mu, sigma) * dt

        if record_step:
            self._speed_history.append(new_speed)
            self._angle_history.append(new_angle)

        return new_speed, new_angle

    @property
    def velocity_history(self):
        """
        Euclidean velocity history
        """
        idx0 = self._velocity_history.shape[0]
        speed = np.array(self._speed_history[idx0:])
        phi = np.array(self._angle_history[idx0:])
        euclidean_direction = np.stack([np.cos(phi), np.sin(phi)], axis=-1)
        velocity = euclidean_direction * speed[..., None]
        self._velocity_history = np.concatenate([self._velocity_history, velocity])
        return self._velocity_history

    @property
    def position_history(self):
        """
        Path integration (Euclidean position) history
        """
        idx0 = self._position_history.shape[0]
        delta_p = np.cumsum(self.velocity_history[idx0:], axis=0)
        self._position_history = np.concatenate(
            [self._position_history, delta_p + self._position_history[-1]]
        )
        return self._position_history

    @property
    def speed_history(self):
        return np.array(self._speed_history)

    @property
    def angle_history(self):
        return np.array(self._angle_history)


if __name__ == "__main__":
    """
    Simple tests/inspection of class methods
    """
    ag = Agent(0, 0)

    for i in range(100000):
        ag.step()

    print(ag.speed_history)
    print(ag.angle_history)
    print(ag.velocity_history)
    print(ag.position_history)
