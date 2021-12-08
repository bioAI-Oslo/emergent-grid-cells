# import matplotlib
# matplotlib.use('nbAgg') # Tried to use same backend as jupyter notebook
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from methods import *


class RatemapDynamics(object):
    def __init__(self, ratemaps, scores, **kwargs):
        """
        ratemaps shape: (S,a**2,resy,resx)
        scores shape: (2,)
        """
        self.ratemaps = ratemaps
        self.scores = scores
        # Setup the figure and axes...
        ncells = int(np.sqrt(ratemaps.shape[1]))
        self.fig, self.axs = plt.subplots(
            nrows=ncells, ncols=ncells, squeeze=False, **kwargs
        )

        # Then setup FuncAnimation.
        self.save_count = ratemaps.shape[0]
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update,
            init_func=self.setup_plot,
            interval=100,  # time in ms between frames
            # repeat_delay=1000, # delay before loop
            blit=False,  # for OSX?
            save_count=self.save_count,  # #frames
            **kwargs,
        )

    def setup_plot(self):
        """Initial plot."""
        multiimshow(self.ratemaps[0], self.axs)
        self.fig.suptitle(f"End grid scores {self.scores[0]} -- {self.scores[1]}")

    def update(self, k):
        # flatten list of Axes and remove previously drawn image
        flat_axs = [ax for axs_1d in self.axs for ax in axs_1d]
        [ax.images.pop(-1) for ax in flat_axs]
        # draw new image
        multiimshow(self.ratemaps[k], self.axs)
