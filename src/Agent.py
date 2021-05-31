import numpy as np
from ctimeit import ctimeit  # custom timing of functions


def batch_generator(batch_size=64,*args,**kwargs):
    """Mini-batch trajectory generator"""
    tgen = trajectory_generator(*args,**kwargs)
    seq_len = args[1] if len(args) >= 2 else kwargs['seq_len']
    mb_pos, mb_vel = np.zeros((batch_size,seq_len,2)), np.zeros((batch_size,seq_len,2))
    
    while True:
        for i in range(batch_size):
            pos,vel = next(tgen)[:2]
            mb_pos[i] = pos
            mb_vel[i] = vel

        yield mb_pos, mb_vel

def trajectory_generator(
    environment, seq_len=20, start_angle=None, start_pos=None, **kwargs
):
    # create agent
    sa, sp = start_angle, start_pos
    agent = Agent(
        np.random.uniform(0, 2 * np.pi), environment.sample_uniform(1), **kwargs
    )

    while True:
        # re-initialize agent
        sa = np.random.uniform(0, 2 * np.pi) if start_angle is None else start_angle
        sp = environment.sample_uniform(1) if start_pos is None else start_pos
        agent.__init__(sa, sp, **kwargs)

        # generate track
        for i in range(seq_len):
            start_pos = environment.sample_uniform(1)
            start_angle = np.random.uniform(0, 2 * np.pi)

            agent.step(environment.avoid_walls)

        yield agent._positions, agent._velocities, agent.hds, agent.speeds, agent.turns, agent


class Agent:
    def __init__(
        self, angle0=None, p0=None, dt=0.02, sigma=5.76 * 2, b=0.13 * 2 * np.pi, mu=0
    ):
        """
        default constants are the ones Sorscher used
        """
        self.dt = dt
        self.sigma = sigma  # stdev rotation velocity (rads/sec)
        self.b = b  # forward velocity rayleigh dist scale (m/sec)
        self.mu = mu  # turn angle bias

        # N+1 len array histories (since we include start pos and hd)
        self.hds = np.array([angle0]) if angle0 is None else np.random.uniform(0, 2 * np.pi,size=1)  # head direction history
        self.speeds = np.zeros(1)  # speed history
        self.turns = np.zeros(1)  # turn direction history
        self._velocities = np.zeros(
            (0, 2)
        )  # velocity history (also N+1, but only when called)
        self._positions = np.zeros((1, 2)) if p0 is None else p0  # position history

        # add size to rat? e.g as an ellipsoid?

    def step(self, avoid_walls, record_step=True):
        """
        Sample a velocity vector - indirectly through speed
        and angle, i.e. (s,phi). The angle is an offset to
        the angle at the previous time step.
        """
        new_speed = np.random.rayleigh(self.b) * self.dt
        new_turn = np.random.normal(self.mu, self.sigma) * self.dt

        new_speed, new_turn = avoid_walls(
            self.positions[-1], self.hds[-1], new_speed, new_turn
        )
        new_hd = np.mod(self.hds[-1] + new_turn, 2 * np.pi)

        if record_step:
            self.speeds = np.append(self.speeds, new_speed)
            self.hds = np.append(self.hds, new_hd)
            self.turns = np.append(self.turns, new_turn)

        return new_speed, new_hd

    @property
    def velocities(self):
        """
        Euclidean velocity history
        """
        idx0 = self._velocities.shape[0]
        if idx0 == self.speeds.shape[0]:
            return self._velocities

        direction = np.stack(
            [np.cos(self.hds[idx0:]), np.sin(self.hds[idx0:])], axis=-1
        )
        velocity = direction * self.speeds[idx0:][..., None]
        self._velocities = np.concatenate([self._velocities, velocity])
        return self._velocities

    @property
    def positions(self):
        """
        Path integration (Euclidean position) history
        """
        idx0 = self._positions.shape[0]
        if idx0 == self.speeds.shape[0]:
            return self._positions

        delta_p = np.cumsum(self.velocities[idx0:], axis=0)
        self._positions = np.concatenate(
            [self._positions, delta_p + self._positions[-1]]
        )
        return self._positions

    def plot_trajectory(self, ax, ds=4):
        # plot animal path
        n=self.positions.shape[0]
        c=np.zeros((n,4))
        c[:,-1] = 1
        c[:,:-1] = 0.9-np.linspace(0,0.9,n)[:,None]
        ax.scatter(*self.positions.T,s=0.1,c=c)

        i = 0
        for pos,vel in zip(self.positions[::ds], self.velocities[::ds]):
            ax.arrow(*pos,*vel,head_width=0.02,color=c[::ds][i])
            i+=1

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
