# -*- coding: utf-8 -*-
"""
Created on 28/01/2020

@author: Alexandre Naaim
"""
import numpy as np
from .utils.vnop_array import vnop_array as vnop_array
from .HomogeneousMatrix import HomogeneousMatrix as HomogeneousMatrix


class spherical_model:

    def __init__(self, segment_distal, segment_proximal, point_distal, point_proximal,
                 name_joint, euler_sequences, point_of_moment_calculus=None, frame_moment=None):
        # Information that will be used later to generate Kinematicschain
        self.name_joint = name_joint
        self.euler_sequences = euler_sequences
        # Point where the forces and moment are calculated
        if point_of_moment_calculus is None:
            nb_frame = segment_distal.u.shape[1]
            X_glob = np.tile(np.array([1, 0, 0])[:, np.newaxis], (1, nb_frame))
            Y_glob = np.tile(np.array([0, 1, 0])[:, np.newaxis], (1, nb_frame))
            Z_glob = np.tile(np.array([0, 0, 1])[:, np.newaxis], (1, nb_frame))
            self.point_of_moment_calculus = HomogeneousMatrix(
                X_glob, Y_glob, Z_glob, point_proximal)
        else:
            self.point_of_moment_calculus = point_of_moment_calculus

        if frame_moment is None:
            self.frame_moment = segment_proximal.Tdist
        else:
            self.frame_moment = frame_moment

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

        # We add the value of the distal and proximal segment to be able to generate
        # automatically build the tree of segment and link.
        self.segment_distal = segment_distal
        self.segment_proximal = segment_proximal
        self.nVdistal = nVdistal
        self.nVproximal = nVproximal
        self.nb_constraint = 3

    def get_phik(self, segment_distal, segment_proximal):
        phik = np.dot(self.nVproximal.T, segment_proximal.Q) - \
            np.dot(self.nVdistal.T, segment_distal.Q)
        return phik

    # On ajoute les paramètre pour que meme si on met des paramètre ca fonctionne
    # Even if the segment_proximal is not used we keep it in the data to facilitate the loop
    # in multi-body optimisation
    def get_Kk(self, segment_distal, segment_proximal):
        nb_frame = segment_distal.u.shape[1]
        Kk = np.zeros((3, 2*12))
        Kk[:, 0:12] = -self.nVdistal.T
        Kk[:, 12:] = self.nVproximal.T
        Kk_final = np.tile(Kk[:, :, np.newaxis], (1, 1, nb_frame))
        return Kk_final

    def get_dKlambdakdQ(self, nb_frame, lambda_k):
        return np.zeros((2*12, 2*12, nb_frame))

    def change_segment(self, segment_distal, segment_proximal, point_of_moment_calculus=None, frame_moment=None):
        self.segment_distal = segment_distal
        self.segment_proximal = segment_proximal
        # Point where the forces and moment are calculated
        if point_of_moment_calculus is None:
            nb_frame = segment_distal.u.shape[1]
            X_glob = np.tile(np.array([1, 0, 0])[:, np.newaxis], (1, nb_frame))
            Y_glob = np.tile(np.array([0, 1, 0])[:, np.newaxis], (1, nb_frame))
            Z_glob = np.tile(np.array([0, 0, 1])[:, np.newaxis], (1, nb_frame))
            self.point_of_moment_calculus = HomogeneousMatrix(
                X_glob, Y_glob, Z_glob, self.segment_proximal.rp)
        else:
            self.point_of_moment_calculus = point_of_moment_calculus

        if frame_moment is None:
            self.frame_moment = segment_proximal.Tdist
        else:
            self.frame_moment = frame_moment


class universal_model:

    def __init__(self, segment_distal, segment_proximal, axe_distal, axe_proximal,
                 name_joint, euler_sequences, point_of_moment_calculus=None, frame_moment=None, theta_1=None):
        # segment_distal : a segment object
        # segment_proximal : a segment object
        # axe_distal : a string designing the distal vector that should be kept at constant angle
        # axe_proximal : a string designing the proximal vector that should be kept at constant angle
        # name_joint : a string containing the name of the joint
        # euler_sequences : a string containing the euler sequences used to calculate kinematics in KinematicsChains
        # point_of_moment_calculus : a HomogeneousMatrix object containing the point where the moment is calculated
        # frame_moment : a HomogeneousMatrix object containing the frame in which the moment should be calculated if
        # the moment in not calculated in the JCS (Joint Coordinate System)

        # Information that will be used later to generate Kinematicschain
        self.name_joint = name_joint
        self.euler_sequences = euler_sequences
        # Point where the forces and moment are calculated
        if point_of_moment_calculus is None:
            nb_frame = segment_distal.u.shape[1]
            X_glob = np.tile(np.array([1, 0, 0])[:, np.newaxis], (1, nb_frame))
            Y_glob = np.tile(np.array([0, 1, 0])[:, np.newaxis], (1, nb_frame))
            Z_glob = np.tile(np.array([0, 0, 1])[:, np.newaxis], (1, nb_frame))
            self.point_of_moment_calculus = HomogeneousMatrix(
                X_glob, Y_glob, Z_glob, segment_proximal.rd)
        else:
            self.point_of_moment_calculus = point_of_moment_calculus

        if frame_moment is None:
            self.frame_moment = segment_proximal.Tdist
        else:
            self.frame_moment = frame_moment

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

        # We add the value of the distal and proximal segment to be able to generate
        # automatically build the tree of segment and link.
        self.segment_distal = segment_distal
        self.segment_proximal = segment_proximal
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

    def change_segment(self, segment_distal, segment_proximal, point_of_moment_calculus=None, frame_moment=None):
        self.segment_distal = segment_distal
        self.segment_proximal = segment_proximal
        # Point where the forces and moment are calculated
        if point_of_moment_calculus is None:
            nb_frame = segment_distal.u.shape[1]
            X_glob = np.tile(np.array([1, 0, 0])[
                             :, np.newaxis], (1, nb_frame))
            Y_glob = np.tile(np.array([0, 1, 0])[
                             :, np.newaxis], (1, nb_frame))
            Z_glob = np.tile(np.array([0, 0, 1])[
                             :, np.newaxis], (1, nb_frame))
            self.point_of_moment_calculus = HomogeneousMatrix(
                X_glob, Y_glob, Z_glob, self.segment_proximal.rp)
        else:
            self.point_of_moment_calculus = point_of_moment_calculus

        if frame_moment is None:
            self.frame_moment = segment_proximal.Tdist
        else:
            self.frame_moment = frame_moment


class hinge_model:

    def __init__(self, segment_distal, segment_proximal,
                 name_joint, euler_sequences, point_of_moment_calculus=None, frame_moment=None,
                 theta_1=None, theta_2=None):
        # Information that will be used later to generate Kinematicschain
        self.name_joint = name_joint
        self.euler_sequences = euler_sequences
        # Point where the forces and moment are calculated
        if point_of_moment_calculus is None:
            nb_frame = segment_distal.u.shape[1]
            X_glob = np.tile(np.array([1, 0, 0])[:, np.newaxis], (1, nb_frame))
            Y_glob = np.tile(np.array([0, 1, 0])[:, np.newaxis], (1, nb_frame))
            Z_glob = np.tile(np.array([0, 0, 1])[:, np.newaxis], (1, nb_frame))
            self.point_of_moment_calculus = HomogeneousMatrix(
                X_glob, Y_glob, Z_glob, segment_proximal.rd)
        else:
            self.point_of_moment_calculus = point_of_moment_calculus

        if frame_moment is None:
            self.frame_moment = segment_proximal.Tdist
        else:
            self.frame_moment = frame_moment

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
        # We add the value of the distal and proximal segment to be able to generate
        # automatically build the tree of segment and link.
        self.segment_distal = segment_distal
        self.segment_proximal = segment_proximal
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

    def change_segment(self, segment_distal, segment_proximal, point_of_moment_calculus=None, frame_moment=None):
        self.segment_distal = segment_distal
        self.segment_proximal = segment_proximal
        # Point where the forces and moment are calculated
        if point_of_moment_calculus is None:
            nb_frame = segment_distal.u.shape[1]
            X_glob = np.tile(np.array([1, 0, 0])[
                             :, np.newaxis], (1, nb_frame))
            Y_glob = np.tile(np.array([0, 1, 0])[
                             :, np.newaxis], (1, nb_frame))
            Z_glob = np.tile(np.array([0, 0, 1])[
                             :, np.newaxis], (1, nb_frame))
            self.point_of_moment_calculus = HomogeneousMatrix(
                X_glob, Y_glob, Z_glob, self.segment_proximal.rp)
        else:
            self.point_of_moment_calculus = point_of_moment_calculus

        if frame_moment is None:
            self.frame_moment = segment_proximal.Tdist
        else:
            self.frame_moment = frame_moment


class constant_distance:

    def __init__(self, segment_distal, segment_proximal, point_distal, point_proximal,
                 name_joint, euler_sequences, point_of_moment_calculus=None, frame_moment=None,
                 distance=None):

        # Information that will be used later to generate Kinematicschain
        self.name_joint = name_joint
        self.euler_sequences = euler_sequences
        # Point where the forces and moment are calculated
        if point_of_moment_calculus is None:
            nb_frame = segment_distal.u.shape[1]
            X_glob = np.tile(np.array([1, 0, 0])[:, np.newaxis], (1, nb_frame))
            Y_glob = np.tile(np.array([0, 1, 0])[:, np.newaxis], (1, nb_frame))
            Z_glob = np.tile(np.array([0, 0, 1])[:, np.newaxis], (1, nb_frame))
            self.point_of_moment_calculus = HomogeneousMatrix(
                X_glob, Y_glob, Z_glob, segment_proximal.rd)
        else:
            self.point_of_moment_calculus = point_of_moment_calculus

        if frame_moment is None:
            self.frame_moment = segment_proximal.Tdist
        else:
            self.frame_moment = frame_moment

        # Calcul matix
        nVdistal = np.zeros((12, 3))
        nVproximal = np.zeros((12, 3))
        if distance is None:
            self.distance = np.mean(np.linalg.norm(
                (point_distal-point_proximal), axis=0))
        else:
            self.distance = distance

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

        # We add the value of the distal and proximal segment to be able to generate
        # automatically build the tree of segment and link.
        self.segment_distal = segment_distal
        self.segment_proximal = segment_proximal
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

    def change_segment(self, segment_distal, segment_proximal, point_of_moment_calculus=None, frame_moment=None):
        self.segment_distal = segment_distal
        self.segment_proximal = segment_proximal
        # Point where the forces and moment are calculated
        if point_of_moment_calculus is None:
            nb_frame = segment_distal.u.shape[1]
            X_glob = np.tile(np.array([1, 0, 0])[
                             :, np.newaxis], (1, nb_frame))
            Y_glob = np.tile(np.array([0, 1, 0])[
                             :, np.newaxis], (1, nb_frame))
            Z_glob = np.tile(np.array([0, 0, 1])[
                             :, np.newaxis], (1, nb_frame))
            self.point_of_moment_calculus = HomogeneousMatrix(
                X_glob, Y_glob, Z_glob, self.segment_proximal.rp)
        else:
            self.point_of_moment_calculus = point_of_moment_calculus

        if frame_moment is None:
            self.frame_moment = segment_proximal.Tdist
        else:
            self.frame_moment = frame_moment


class no_model:

    def __init__(self, segment_distal, segment_proximal,
                 name_joint, euler_sequences, point_of_moment_calculus=None, frame_moment=None):
        # Information that will be used later to generate Kinematicschain
        self.name_joint = name_joint
        self.euler_sequences = euler_sequences
        # Point where the forces and moment are calculated
        if point_of_moment_calculus is None:
            nb_frame = segment_distal.u.shape[1]
            X_glob = np.tile(np.array([1, 0, 0])[:, np.newaxis], (1, nb_frame))
            Y_glob = np.tile(np.array([0, 1, 0])[:, np.newaxis], (1, nb_frame))
            Z_glob = np.tile(np.array([0, 0, 1])[:, np.newaxis], (1, nb_frame))
            self.point_of_moment_calculus = HomogeneousMatrix(
                X_glob, Y_glob, Z_glob, segment_proximal.rd)
        else:
            self.point_of_moment_calculus = point_of_moment_calculus

        if frame_moment is None:
            self.frame_moment = segment_proximal.Tdist
        else:
            self.frame_moment = frame_moment

        # Calcul matix
        self.nb_constraint = 0
        # We add the value of the distal and proximal segment to be able to generate
        # automatically build the tree of segment and link.
        self.segment_distal = segment_distal
        self.segment_proximal = segment_proximal

    def get_phik(self, segment_distal, segment_proximal):
        nb_frame = segment_distal.u.shape[1]
        return np.zeros((0, nb_frame))

    def get_Kk(self, segment_distal, segment_proximal):
        nb_frame = segment_distal.u.shape[1]
        Kk_final = np.zeros((0, 2*12, nb_frame))
        return Kk_final

    def get_dKlambdakdQ(self, nb_frame, lambda_k):
        return np.zeros((2*12, 2*12, nb_frame))

    def change_segment(self, segment_distal, segment_proximal, point_of_moment_calculus=None, frame_moment=None):
        self.segment_distal = segment_distal
        self.segment_proximal = segment_proximal
        # Point where the forces and moment are calculated
        if point_of_moment_calculus is None:
            nb_frame = segment_distal.u.shape[1]
            X_glob = np.tile(np.array([1, 0, 0])[
                             :, np.newaxis], (1, nb_frame))
            Y_glob = np.tile(np.array([0, 1, 0])[
                             :, np.newaxis], (1, nb_frame))
            Z_glob = np.tile(np.array([0, 0, 1])[
                             :, np.newaxis], (1, nb_frame))
            self.point_of_moment_calculus = HomogeneousMatrix(
                X_glob, Y_glob, Z_glob, self.segment_proximal.rp)
        else:
            self.point_of_moment_calculus = point_of_moment_calculus

        if frame_moment is None:
            self.frame_moment = segment_proximal.Tdist
        else:
            self.frame_moment = frame_moment
