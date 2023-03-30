#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh

def plot_gridmap(gridmap):
    plt.figure()
    plt.imshow(gridmap, cmap='Greys',vmin=0, vmax=1)
    
def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size/res)), int(np.ceil(size/res))])
    return gridmap

def world2map(pose, gridmap, map_res):
    origin = np.array(gridmap.shape)/2
    new_pose = np.zeros_like(pose)
    new_pose[:,0] = np.round(pose[:,0]/map_res) + origin[0];
    new_pose[:,1] = np.round(pose[:,1]/map_res) + origin[1];
    return new_pose.astype(int)

def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr    

def ranges2points(ranges):
    # laser properties
    start_angle = -1.5708
    angular_res = 0.0087270
    max_range = 30
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    angles = np.linspace(start_angle, start_angle + (num_beams*angular_res), num_beams)[idx]
    points = np.array([np.multiply(ranges[idx], np.cos(angles)), np.multiply(ranges[idx], np.sin(angles))])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom

def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # covert to map frame
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2,:]
    return m_points

def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    w_pose = np.array([w_pose])
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose  

def bresenham(x0, y0, x1, y1):
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l
    
def prob2logodds(p):
    return np.log(p/(1-p))
    
def logodds2prob(l):
    return 1 / (np.exp(-l) + 1)
    
def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
    return prob_occ if np.all(cell==endpoint) else prob_free
    
def grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap, map_res, prob_occ, prob_free, prior):
    for ranges,pose in zip(ranges_raw,poses_raw):
        footpoint = poses2cells(pose,occ_gridmap,map_res)[0]
        endpoints = ranges2cells(ranges,pose,occ_gridmap,map_res)
        for i in range(endpoints.shape[0]):
            for cell in bresenham(footpoint[0],footpoint[1],endpoints[0,i],endpoints[1,i]):
                occ_gridmap[cell[0],cell[1]] += prob2logodds(inv_sensor_model(cell, endpoints[:,i].reshape(2), prob_occ, prob_free))
