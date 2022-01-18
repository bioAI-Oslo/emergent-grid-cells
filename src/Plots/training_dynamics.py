import numpy as np
import torch
import matplotlib.pyplot as plt


def training_dynamics(experiment, model, ps=[0.05, 0.5, 0.95]):
    """
    3 x len(environments)+1 plot matrix:
    first row: total loss history over different enviroments
    second row: prediction error
    third row: L2-regularization

    The three training time sample points:
    1. ps[0] reduction of (minimum - maximum Cuba-Libre metric)
    2. ps[1] reduction of (minimum - maximum Cuba-Libre metric)
    3. ps[2] reduction of (minimum - maximum Cuba-Libre metric)
    """
    
    fig, axs = plt.subplots(nrows=3, ncols=len(experiment.environments) + 1)
    
    return None
