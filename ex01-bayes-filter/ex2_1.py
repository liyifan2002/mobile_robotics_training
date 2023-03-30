#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_belief(belief):
    
    plt.figure()
    
    ax = plt.subplot(2,1,1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0],1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")
    
    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")


def motion_model(action, belief):
    move_dir = 1 if action == "F" else -1
    new_belief = np.zeros_like(belief)
    for i in range(len(belief)):
        belief_correct = belief[i - move_dir] if 0 <= i - move_dir < len(belief) else 0
        belief_opposite = belief[i + move_dir] if 0 <= i + move_dir < len(belief) else 0
        new_belief[i] = 0.7 *belief_correct + 0.2 *belief[i] + 0.1 * belief_opposite
    
    return new_belief

def sensor_model(observation, belief, world):
    for i in range(len(belief)):
        if(world[i] == 1):
            belief[i] *= 0.7 if observation == 1 else 0.3
        if(world[i] == 0):
            belief[i] *= 0.9 if observation == 0 else 0.1
    return belief/belief.sum()

def recursive_bayes_filter(actions, observations, belief, world):
    for step in range(len(actions)):
        belief = motion_model(actions[step],belief)
        belief = sensor_model(observations[step],belief,world)
    return belief
