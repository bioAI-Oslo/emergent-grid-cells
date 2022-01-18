import numpy as np
import torch
import matplotlib.pyplot as plt

import sys

sys.path.append("../")

from methods import filenames, multiimshow

def baround(x, base, *args, **kwargs):
    """
    Round to nearest base. e.g. 
    10 -> 10
    12 -> 10
    13 -> 15

    Taken from:
    https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
    """
    return base * np.around(x / base, *args, **kwargs)

def training_dynamics_default(
    experiment, loss_history, training_metrics, ps=[0.95, 0.5, 0.05]
):
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
    # fig, axs = plt.subplots(nrows=3, ncols=len(experiment.environments) + 1)
    fig = plt.figure()
    grid = plt.GridSpec(6, 2*len(experiment.environments) + 1)

    # plot loss history
    ax00 = fig.add_subplot(grid[0:2, 0:2])
    ax00.plot(loss_history, label="Loss")
    ax00.plot(training_metrics["CE"], label="CE")
    ax00.plot(training_metrics["entropy"], label="Entropy")
    ax00.set_title("Loss History")
    ax00.legend()

    # plot L2-penalty
    ax10 = fig.add_subplot(grid[2:4, 0:2])
    ax10.plot(training_metrics["l2_reg"])
    ax10.set_title("L2-Penalty")
    ax10.axhline(0, ls=":")

    # plot Euclidean prediction error
    ax20 = fig.add_subplot(grid[4:6, 0:2])
    ax20.plot(training_metrics["pred_error"], label="Prediction")
    ax20.plot(training_metrics["true_error"], label="Label")
    ax20.set_title("Decoding Error")
    ax20.legend()

    # find training phases as a fraction of loss-differences
    kl = np.array(training_metrics["KL"])
    kl_diff = np.max(kl) - np.min(kl)
    kl_phases = kl_diff * np.array(ps)
    kl_phase_idxs = kl_phases[:, None] - kl[None]  # shape: (3, #epochs)
    kl_phase_idxs = np.argmin(kl_phase_idxs, axis=-1)  # shape: (3,)

    # find closest (rounded) model checkpoints to the phase idxs
    checkpoint_filenames = filenames(experiment.paths['checkpoints'])
    checkpoint_nums = [int(checkpoint_filename) for checkpoint_filename in checkpoint_filenames]
    save_freq = checkpoint_nums[2] - checkpoint_nums[1]
    checkpoint_phase_nums = baround(kl_phase_idxs, save_freq)
    checkpoint_phase_filenames = [f"{checkpoint_phase_num:04d}" for checkpoint_phase_num in checkpoint_phase_nums]

    # load grid scores for environments
    sorted_scores = []
    sort_idxs = []
    for grid_score_filename in filenames(experiment.paths['grid_scores']):
        with open(grid_score_filename, 'rb') as f:
            score_map = pickle.load(f)
        
        sort_idxs.append(np.argsort(score_map)[::-1])
        sorted_scores.append(score_map[sort_idxs[-1]])


    # load ratemaps for environment
    ratemaps = []
    for j, checkpoint_phase_filename in enumerate(checkpoint_phase_filenames):
        for env_i in range(len(experiment.environments)):
            ratemap_filename = experiment.paths['ratemaps'] / f"env_{env_i}" / checkpoint_phase_filename 
            with open(ratemap_filename, 'rb') as f:
                ratemap = pickle.load(f)
            
            # select best grid cell from env_i
            ratemap[sort_idxs[env_i]][0]
            
            # select correct 'subplotaxes' and plot ratemap there
            ax = fig.add_subplot(grid[2*j + (env_i // 2), 2 + (env_i % 2)])
            ax.imshow(ratemap, cmap='jet')
        




    return None
