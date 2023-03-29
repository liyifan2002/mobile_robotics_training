#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

thr_free = 0.9

def plot_path(path, x_start, x_goal, M):
    plt.matshow(M, cmap="gray")
    if path.shape[0] > 2:
        plt.plot(path[:, 1], path[:, 0], 'b')
    plt.plot(x_start[1], x_start[0], 'or')
    plt.plot(x_goal[1], x_goal[0], 'xg')
    plt.show()


def is_valid(v):
    if v > thr_free:
        return True
    return False

# def plan_path_uninformed(x_start, x_goal, M):
	# add code here
    
# def plan_path_astar(x_start, x_goal, M):
	# add code here
  
