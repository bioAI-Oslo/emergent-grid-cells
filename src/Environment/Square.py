import numpy as np

from Environment import Environment


class Square(Environment):
    def __init__(self, boxsize=(1, 1), res=(32, 32)):
        self.origo = (0, 0)
        self.boxsize = boxsize
        self.res = res

        # initialize board
        xx, yy = np.meshgrid(
            np.linspace(self.origo[0], boxsize[0], res[0]),
            np.linspace(self.origo[1], boxsize[0], res[0]),
        )
        self.board = np.stack([xx, yy], axis=-1)

    def uniform_sample(self, ns=1):
        """
        Sampling a 2d-square is trivial with numpy
        """
        return np.random.uniform(self.origo, self.boxsize, size=(ns, 2))

    def walls(self,positions,velocities):
        

if __name__ == "__main__":
    print(Square())
