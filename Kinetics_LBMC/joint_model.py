# -*- coding: utf-8 -*-
"""
Created on 28/01/2020

@author: Alexandre Naaim
"""
import numpy as np
from .utils.vnop_array import vnop_array as vnop_array


class spherical_model:

    def __init__(self, segment_distal, segment_proximal, point_distal, point_proximal):
        # Calcul matix
        nVdistal = np.zeros((12, 3))
        nVproximal = np.zeros((12, 3))

        nv_temp_distal = np.mean(vnop_array(point_distal-segment_distal.rp,
                                            segment_distal.u,
                                            (segment_distal.rp-segment_distal.rd),
                                            segment_distal.w), axis=1)

        nv_temp_proximal = np.mean(vnop_array(point_proximal-segment_proximal.rp,
                                              segment_proximal.u,
                                              (segment_proximal.rp -
                                               segment_proximal.rd),
                                              segment_proximal.w), axis=1)

        nVdistal[0:3, :] = nv_temp_distal[0]*np.eye(3)
        nVdistal[3:6, :] = (1+nv_temp_distal[1])*np.eye(3)
        nVdistal[6:9, :] = -nv_temp_distal[1]*np.eye(3)
        nVdistal[9:12, :] = nv_temp_distal[2]*np.eye(3)

        nVproximal[0:3, :] = nv_temp_proximal[0]*np.eye(3)
        nVproximal[3:6, :] = (1+nv_temp_proximal[1])*np.eye(3)
        nVproximal[6:9, :] = -nv_temp_proximal[1]*np.eye(3)
        nVproximal[9:12, :] = nv_temp_proximal[2]*np.eye(3)

        self.nVdistal = nVdistal
        self.nVproximal = nVproximal
        self.nb_constraint = 3

    def get_phik(self, segment_distal, segment_proximal):
        phik = np.dot(self.nVproximal.T, segment_proximal.Q) - \
            np.dot(self.nVdistal.T, segment_distal.Q)
        return phik

    # On ajoute les paramètre pour que meme si on met des paramètre ca fonctionne
    def get_Kk(self, segment_distal, segment_proximal):
        nb_frame = segment_distal.u.shape[1]
        Kk = np.zeros((3, 2*12))
        Kk[:, 0:12] = -self.nVdistal.T
        Kk[:, 12:] = self.nVproximal.T
        Kk_final = np.tile(Kk[:, :, np.newaxis], (1, 1, nb_frame))
        return Kk_final

    def get_dKlambdakdQ(self, nb_frame, lambda_k):
        return np.zeros((2*12, 2*12, nb_frame))


class universal_model:

    def __init__(self, segment_distal, segment_proximal, axe_distal,
                 axe_proximal, theta_1=None):
        # les axes peuvent être u v w
        if axe_distal == 'u':
            axe_dist_calc = segment_distal.u
        elif axe_distal == 'v':
            axe_dist_calc = (segment_distal.rp -
                             segment_distal.rd)/segment_distal.length_mean
        elif axe_distal == 'w':
            axe_dist_calc = axe_distal.w

        if axe_proximal == 'u':
            axe_prox_calc = segment_proximal.u
        elif axe_proximal == 'v':
            axe_prox_calc = (segment_proximal.rp -
                             segment_proximal.rd)/segment_proximal.length_mean
        elif axe_proximal == 'w':
            axe_prox_calc = segment_proximal.w

        if theta_1 is None:
            self.theta_1 = np.mean(
                np.arccos(np.sum(axe_prox_calc*axe_dist_calc, axis=0)))
        else:
            self.theta_1 = theta_1

        self.axe_distal = axe_distal
        self.axe_proximal = axe_proximal

        self.nb_constraint = 4

    def get_phik(self, segment_distal, segment_proximal):
        phik = np.zeros((4, segment_proximal.u.shape[1]))
        phik[0:3, :] = (segment_proximal.rd-segment_distal.rp)

        coeff = np.ones((1, segment_proximal.u.shape[1]))

        if self.axe_distal == 'u':
            axe_dist_calc = segment_distal.u
        elif self.axe_distal == 'v':
            axe_dist_calc = (segment_distal.rp - segment_distal.rd)
            coeff *= segment_distal.length_mean
        elif self.axe_distal == 'w':
            axe_dist_calc = segment_distal.w

        if self.axe_proximal == 'u':
            axe_prox_calc = segment_proximal.u
        elif self.axe_proximal == 'v':
            axe_prox_calc = (segment_proximal.rp - segment_proximal.rd)
            coeff *= segment_proximal.length_mean
        elif self.axe_proximal == 'w':
            axe_prox_calc = segment_proximal.w

        phik[3, :] = np.sum(axe_dist_calc*(axe_prox_calc),
                            0) - np.cos(self.theta_1)*coeff

        return phik

    def get_Kk(self, segment_distal, segment_proximal):
        Kk = np.zeros((4, 2*12, segment_distal.u.shape[1]))
        Kk[0:3, 3:6, :] = -np.eye(3)[:, :, np.newaxis]
        Kk[0:3, 12+6:12+9, :] = np.eye(3)[:, :, np.newaxis]

        if self.axe_distal == 'u':
            axe_dist_calc = segment_distal.u
        elif self.axe_distal == 'v':
            axe_dist_calc = (segment_distal.rp - segment_distal.rd)
        elif self.axe_distal == 'w':
            axe_dist_calc = segment_distal.w

        if self.axe_proximal == 'u':
            axe_prox_calc = segment_proximal.u
        elif self.axe_proximal == 'v':
            axe_prox_calc = (segment_proximal.rp - segment_proximal.rd)
        elif self.axe_proximal == 'w':
            axe_prox_calc = segment_proximal.w

        if self.axe_distal == 'u':
            Kk[3, 0:3, :] = axe_prox_calc
        elif self.axe_distal == 'v':
            Kk[3, 3:6, :] = axe_prox_calc
            Kk[3, 6:9, :] = -axe_prox_calc
        elif self.axe_distal == 'w':
            Kk[3, 9:12, :] = axe_prox_calc

        if self.axe_proximal == 'u':
            Kk[12:15, :, :] = axe_dist_calc
        elif self.axe_proximal == 'v':
            Kk[15:18, :, :] = axe_dist_calc
            Kk[18:21, :, :] = -axe_dist_calc
        elif self.axe_proximal == 'w':
            Kk[21:24, :, :] == axe_dist_calc

        return Kk

    def get_dKlambdakdQ(self, nb_frame, lambda_k):
        diag_3 = np.identity(3)[:, :, np.newaxis]
        dKlambdakdQ = np.zeros((2*12, 2*12, lambda_k.shape[1]))
        # List must be defined to be iterable
        if self.axe_distal == 'u':
            distal = [0]
        elif self.axe_distal == 'v':
            distal = [1, -2]
        elif self.axe_distal == 'w':
            distal = [3]

        if self.axe_proximal == 'u':
            proximal = [0]
        elif self.axe_proximal == 'v':
            proximal = [1, -2]
        elif self.axe_proximal == 'w':
            proximal = [3]

        value = lambda_k[3, :]*diag_3
        for distal_temp in distal:
            for proximal_temp in proximal:
                dKlambdakdQ[abs(distal_temp)*3:(abs(distal_temp)+1)*3,
                            abs(proximal_temp)*3+12:(abs(proximal_temp)+1)*3+12,
                            :] = np.sign(proximal_temp)*np.sign(distal_temp) * value

        for proximal_temp in proximal:
            for distal_temp in distal:
                dKlambdakdQ[abs(proximal_temp)*3:(abs(proximal_temp)+1)*3,
                            abs(distal_temp)*3+12:(abs(distal_temp)+1)*3+12,
                            :] = np.sign(proximal_temp)*np.sign(distal_temp) * value

        return dKlambdakdQ


class hinge_model:

    def __init__(self, segment_distal, segment_proximal, theta_1=None, theta_2=None):

        if theta_1 is None:
            self.theta_1 = np.mean(np.arccos(
                np.sum(segment_proximal.w *
                       (segment_distal.rp-segment_distal.rd), axis=0)
                / segment_distal.length_mean))
        else:
            self.theta_1 = theta_1
        if theta_2 is None:
            self.theta_2 = np.mean(np.arccos(
                np.sum(segment_proximal.w*segment_distal.u, axis=0)))
        else:
            self.theta_2 = theta_2

        self.nb_constraint = 5

    def get_phik(self, segment_distal, segment_proximal):
        phik = np.zeros((5, segment_proximal.u.shape[1]))
        phik[0:3, :] = (segment_proximal.rd-segment_distal.rp)
        phik[3, :] = np.sum(segment_proximal.w*(segment_distal.rp-segment_distal.rd),
                            0) - segment_distal.length_mean*np.cos(self.theta_1)
        phik[4, :] = np.sum(segment_proximal.w*(segment_distal.u), 0)
        - np.cos(self.theta_2)

        return phik

    def get_Kk(self, segment_distal, segment_proximal):
        Kk = np.zeros((5, 2*12, segment_distal.u.shape[1]))
        Kk[0:3, 3:6, :] = -np.eye(3)[:, :, np.newaxis]
        Kk[0:3, 12+6:12+9, :] = np.eye(3)[:, :, np.newaxis]

        Kk[3, 3:6, :] = segment_proximal.w
        Kk[3, 6:9, :] = -segment_proximal.w
        Kk[3, 12+9:12+12, :] = segment_distal.rp-segment_distal.rd

        Kk[4, 0:3, :] = segment_proximal.w
        Kk[4, 12+9:12+12, :] = segment_distal.u

        return Kk

    def get_dKlambdakdQ(self, nb_frame, lambda_k):
        diag_3 = np.identity(3)[:, :, np.newaxis]
        dKlambdakdQ = np.zeros((2*12, 2*12, lambda_k.shape[1]))

        dKlambdakdQ[0:3, 21:24, :] = lambda_k[4, :]*diag_3

        dKlambdakdQ[3:6, 21:24, :] = lambda_k[3, :]*diag_3

        dKlambdakdQ[6:9, 21:24, :] = -lambda_k[3, :]*diag_3

        dKlambdakdQ[21:24, 0:3, :] = lambda_k[4, :]*diag_3
        dKlambdakdQ[21:24, 3:6, :] = lambda_k[3, :]*diag_3
        dKlambdakdQ[21:24, 6:9, :] = -lambda_k[3, :]*diag_3

        return dKlambdakdQ


class constant_distance:

    def __init__(self, segment_distal, segment_proximal, point_distal, point_proximal):
        # Calcul matix
        nVdistal = np.zeros((12, 3))
        nVproximal = np.zeros((12, 3))

        self.distance = np.mean(np.linalg.norm(
            (point_distal-point_proximal), axis=0))

        nv_temp_distal = np.mean(vnop_array(point_distal-segment_distal.rp,
                                            segment_distal.u,
                                            (segment_distal.rp-segment_distal.rd),
                                            segment_distal.w), axis=1)

        nv_temp_proximal = np.mean(vnop_array(point_proximal-segment_proximal.rp,
                                              segment_proximal.u,
                                              (segment_proximal.rp -
                                               segment_proximal.rd),
                                              segment_proximal.w), axis=1)

        nVdistal[0:3, :] = nv_temp_distal[0]*np.eye(3)
        nVdistal[3:6, :] = (1+nv_temp_distal[1])*np.eye(3)
        nVdistal[6:9, :] = -nv_temp_distal[1]*np.eye(3)
        nVdistal[9:12, :] = nv_temp_distal[2]*np.eye(3)

        nVproximal[0:3, :] = nv_temp_proximal[0]*np.eye(3)
        nVproximal[3:6, :] = (1+nv_temp_proximal[1])*np.eye(3)
        nVproximal[6:9, :] = -nv_temp_proximal[1]*np.eye(3)
        nVproximal[9:12, :] = nv_temp_proximal[2]*np.eye(3)

        self.nVdistal = nVdistal
        self.nVproximal = nVproximal
        self.nb_constraint = 1

    def get_phik(self, segment_distal, segment_proximal):
        phik_temp = (np.dot(self.nVproximal.T, segment_proximal.Q) -
                     np.dot(self.nVdistal.T, segment_distal.Q))
        phik = np.linalg.norm(phik_temp, axis=0)**2-self.distance**2
        return phik

    # On ajoute les paramètre pour que meme si on met des paramètre ca fonctionne
    def get_Kk(self, segment_distal, segment_proximal):
        nb_frame = segment_distal.u.shape[1]
        Kk = np.zeros((3, 2*12))
        Kk[:, 0:12] = -2(np.dot(self.nVproximal.T, segment_proximal.Q) -
                         np.dot(self.nVdistal.T, segment_distal.Q))  # (self.nVdistal.T)
        Kk[:, 12:] = self.nVproximal.T
        Kk_final = np.tile(Kk[:, :, np.newaxis], (1, 1, nb_frame))
        return Kk_final

    def get_dKlambdakdQ(self, nb_frame, lambda_k):
        return np.zeros((2*12, 2*12, nb_frame))


class no_model:

    def __init__(self):
        # Calcul matix
        self.nb_constraint = 0

    def get_phik(self, segment_distal, segment_proximal):
        nb_frame = segment_distal.u.shape[1]
        return np.zeros((0, nb_frame))

    def get_Kk(self, segment_distal, segment_proximal):
        nb_frame = segment_distal.u.shape[1]
        Kk_final = np.zeros((0, 2*12, nb_frame))
        return Kk_final

    def get_dKlambdakdQ(self, nb_frame, lambda_k):
        return np.zeros((2*12, 2*12, nb_frame))
