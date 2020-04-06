# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:18:11 2019

@author: AdminXPS
"""
import numpy as np


def points_treatment_validation_QUT(acq):
    points_temp = acq['data']['points'][0:3, :, :]
    points = np.zeros_like(points_temp)

    # Respect ISB convention
    points[0] = points_temp[0]/1000
    points[1] = points_temp[2]/1000
    points[2] = -points_temp[1]/1000
    # Pas de filtre sinon il y a des différences
    points_treated = points
    return points_treated


def points_treatment_validation_foot(acq):
    points_temp = acq['data']['points'][0:3, :, :]
    points = np.zeros_like(points_temp)

    # Respect ISB convention
    points[0] = points_temp[0]
    points[1] = points_temp[2]
    points[2] = -points_temp[1]
    # Pas de filtre sinon il y a des différences
    points_treated = points
    return points_treated
