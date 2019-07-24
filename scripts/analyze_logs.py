from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

NUM_TEACHER_DEMOS = 100

def plot_returns(returns):
    plt.plot(returns)
    print(len(returns))
    plt.xlabel('Iteration')
    plt.ylabel("Return")
    plt.title("Value Function Only")
    plt.ylim(0, 110)
    plt.savefig('returns.png')
    plt.show()
    plt.close()

def plot_observation_trajs(observations):
    print(len(observations))
    for i in range(len(observations)):
        if i % (len(observations)//10) == 0:
            obs = observations[i]
            plt.plot(obs[:, 0], obs[:, 2], '->', label = "Trajectory at Iteration: " + str(i))

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc='best') 
    plt.savefig('observation_trajs.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    def get_stats(data):
        mu = np.mean(data, axis=0)
        lb = mu - np.std(data, axis=0)
        ub = mu + np.std(data, axis=0)
        return mu, lb, ub

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import colorConverter as cc
    import numpy as np
     
    def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
        # plot the shaded range of the confidence intervals
        plt.fill_between(range(mean.shape[0]), ub, lb,
                         color=color_shading, alpha=.5)
        # plot the mean on top
        plt.plot(mean, color_mean)
    
    ours_1 = sio.loadmat('log/2019-07-23--17:05:37/logs.mat')['returns'][0]
    ours_1[ours_1 > 100] = 100
    # ours_1 = ours_1[212:] # Only show post curriculum
    ours_2 = sio.loadmat('log/2019-07-23--17:05:41/logs.mat')['returns'][0]
    ours_2[ours_2 > 100] = 100
    # ours_2 = ours_2[212:] # Only show post curriculum
    ours_3 = sio.loadmat('log/2019-07-23--17:05:33/logs.mat')['returns'][0]
    ours_3[ours_3 > 100] = 100
    # ours_3 = ours_3[212:] # Only show post curriculum
    ours = [ours_1, ours_2, ours_3]

    pets_1 = sio.loadmat('log/2019-07-18--19:06:37/logs.mat')['returns'][0]
    pets_1[pets_1 > 100] = 100
    # pets_1 = pets_1[212:] # Only show post curriculum
    pets_2 = sio.loadmat('log/2019-07-18--19:05:33/logs.mat')['returns'][0]
    pets_2[pets_2 > 100] = 100
    # pets_2 = pets_2[212:] # Only show post curriculum
    pets_3 = sio.loadmat('log/2019-07-18--19:06:13/logs.mat')['returns'][0]
    pets_3[pets_3 > 100] = 100
    # pets_3 = pets_3[212:] # Only show post curriculum
    pets = [pets_1, pets_2, pets_3]

    petsfd_1 = sio.loadmat('log/2019-02-19--23:59:03/logs.mat')['returns'][0]
    petsfd_1[petsfd_1 > 100] = 100
    # petsfd_1 = petsfd_1[212:] # Only show post curriculum
    petsfd_2 = sio.loadmat('log/2019-02-19--23:58:41/logs.mat')['returns'][0]
    petsfd_2[petsfd_2 > 100] = 100
    # petsfd_2 = petsfd_2[212:] # Only show post curriculum
    petsfd_3 = sio.loadmat('log/2019-02-19--23:57:23/logs.mat')['returns'][0]
    petsfd_3[petsfd_3 > 100] = 100
    # petsfd_3 = petsfd_3[212:] # Only show post curriculum
    petsfd = [petsfd_1, petsfd_2, petsfd_3]

    clone_1 = sio.loadmat('log/2019-07-21--21:58:20/logs.mat')['returns'][0]
    clone_1[clone_1 > 100] = 100
    # clone_1 = clone_1[212: ] # Only show post curriculum
    clone_2 = sio.loadmat('log/2019-07-21--21:58:25/logs.mat')['returns'][0]
    clone_2[clone_2 > 100] = 100
    # clone_2 = clone_2[212: ] # Only show post curriculum
    clone_3 = sio.loadmat('log/2019-07-21--21:58:24/logs.mat')['returns'][0]
    clone_3[clone_3 > 100] = 100
    # clone_3 = clone_3[212: ] # Only show post curriculum
    clone = [clone_1, clone_2, clone_3]
    # generate 3 sets of random means and confidence intervals to plot
    mean0, lb0, ub0 = get_stats(ours)
    mean1, lb1, ub1 = get_stats(pets)
    mean2, lb2, ub2 = get_stats(petsfd)
    mean3, lb3, ub3 = get_stats(clone)

    # plot the data
    fig = plt.figure(1, figsize=(7, 2.5))
    plot_mean_and_CI(mean0, ub0, lb0, color_mean='k', color_shading='k')
    plot_mean_and_CI(mean1, ub1, lb1, color_mean='b', color_shading='b')
    plot_mean_and_CI(mean2, ub2, lb2, color_mean='g--', color_shading='g')
    plot_mean_and_CI(mean3, ub3, lb3, color_mean='r--', color_shading='r')
     
    class LegendObject(object):
        def __init__(self, facecolor='red', edgecolor='white', dashed=False):
            self.facecolor = facecolor
            self.edgecolor = edgecolor
            self.dashed = dashed
     
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height
            patch = mpatches.Rectangle(
                # create a rectangle that is filled with color
                [x0, y0], width, height, facecolor=self.facecolor,
                # and whose edges are the faded color
                edgecolor=self.edgecolor, lw=3)
            handlebox.add_artist(patch)
     
            # if we're creating the legend for a dashed line,
            # manually add the dash in to our rectangle
            if self.dashed:
                patch1 = mpatches.Rectangle(
                    [x0 + 2*width/5, y0], width/5, height, facecolor=self.edgecolor,
                    transform=handlebox.get_transform())
                handlebox.add_artist(patch1)
     
            return patch
     
    bg = np.array([1, 1, 1])  # background of the legend is white
    colors = ['black', 'blue', 'green', 'red']
    # with alpha = .5, the faded color is the average of the background and color
    colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]
     
    plt.legend([0, 1, 2, 3], ['SAVED', 'PETS', 'PETSfD', 'Clone'],
               handler_map={
                   0: LegendObject(colors[0], colors_faded[0]),
                   1: LegendObject(colors[1], colors_faded[1]),
                   2: LegendObject(colors[2], colors_faded[2], dashed=True),
                   3: LegendObject(colors[3], colors_faded[3], dashed=True),
                }, loc='upper right')
     
    plt.title('Pusher Task: Iteration Cost vs. Time Post Curriculum')
    plt.ylabel("Iteration Cost")
    plt.xlabel("Iteration")
    plt.tight_layout()
    plt.grid()
    plt.savefig("pusher.pdf")
    # plt.show()