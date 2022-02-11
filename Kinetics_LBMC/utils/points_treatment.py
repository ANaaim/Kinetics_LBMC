# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:18:11 2019

@author: AdminXPS
"""
import numpy as np
from scipy import signal, interpolate


def points_treatment(acq, fc_marker, unit_point='mm', vector_equivalent=[0, 2, 1], vector_sign=[1, 1, -1]):
    # acq = acqusition file in ezc3d
    # fc = cutoff frequency
    # unit_point unit in which the point are given
    # vector equivalent = which direction the new vector is supposed to be
    # vector_sign = the new sign of the vector
    # Example in ISB Y is pointing upward and in typical c3d file the Z is pointing upward as a result
    # XYZ (0 1 2) become XZ-Y (0 2 -1) which result in vector_equivalent=[0,2,1] and vector_sign=[1,1,-1]
    # By default it is the transition from normal c3d file to ISB

    points_temp = acq['data']['points'][0:3, :, :]
    points = np.zeros_like(points_temp)

    # Respect ISB convention
    if unit_point == 'mm':
        unit_conv = 1000
    elif unit_point == 'm':
        unit_conv = 1
    else:
        print('unit_point in point_treatment is not supported')

    points[0] = vector_sign[0]*points_temp[vector_equivalent[0]]/unit_conv
    points[1] = vector_sign[1]*points_temp[vector_equivalent[1]]/unit_conv
    points[2] = vector_sign[2]*points_temp[vector_equivalent[2]]/unit_conv

    nb_frame = points.shape[2]
    frq_acq = acq['parameters']['POINT']['RATE']['value'][0]

    order_marker = 4.0
    if points.shape[2] > 15:
        x = np.arange(0, nb_frame)
        f = interpolate.interp1d(x, points, axis=2)
        points_interp = f(tuple(x))
        # Filtering
        b, a = signal.butter(order_marker, fc_marker/(0.5*frq_acq))
        points_treated = signal.filtfilt(b, a, points_interp, axis=2)
    else:
        points_treated = points

    return points_treated
