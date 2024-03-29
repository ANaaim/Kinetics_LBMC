from .utils.vnop_array import vnop_array as vnop_array
from .utils.inertia_matrix import dumas
from .HomogeneousMatrix import HomogeneousMatrix
import numpy as np
# -*- coding: utf-8 -*-
"""Segment class used to define anatomical segment based on natural coordinate.

Created on Wed Feb 20 13:21:05 2019

@author: Alexandre Naaïm
"""

# TODO : Add the external forces that are exerted on the solide ==> Prb should be given in global frame
# if not during a MBO there is a risk that the position of the forces might be changed


class Segment:
    # TODO add the name of the marker in the segment to be able to work on it
    def __init__(self, u, rp, rd, w, rm, weight_rm,
                 Btype_prox, Btype_dist,
                 segment_name, rm_name, sexe='M', weight=0, phi_ext=None,
                 segment_static=None, rigid_parameter=False, inertia='dumas',
                 nm_list=None, frame_prox=None):
        self.segment_name = segment_name
        # Q vector parameters
        self.u = np.copy(u)
        self.rp = np.copy(rp)
        self.rd = np.copy(rd)
        self.w = np.copy(w)
        Q = np.zeros((12, u.shape[1]))
        Q[0:3] = np.copy(u)
        Q[3:6] = np.copy(rp)
        Q[6:9] = np.copy(rd)
        Q[9:12] = np.copy(w)
        self.Q = Q
        # Frame number (second dimension of the vector u/rp/rd/w)
        nb_frame = u.shape[1]

        # Point associated to the segment
        self.rm = rm
        self.weight_rm = weight_rm
        # TODO : test if weight_rm is same lenght as rm
        self.rm_name = rm_name
        # Faire different lenght, alpha et beta....
        # Les valeurs uniques sont pour les optimisations
        # Les valeurs calculer pour le segment dans son état est pour la cinématique

        # TODO : Create a constructor where these parameters are given
        self.length = np.sqrt(np.sum((rp - rd)**2, axis=0))
        self.alpha = np.arccos(np.sum((rp - rd)*w, axis=0)/self.length)
        self.beta = np.arccos(np.sum(u*w, axis=0))
        self.gamma = np.arccos(np.sum(u*(rp-rd), axis=0)/self.length)

        if segment_static is None:
            if rigid_parameter:
                self.length_mean = np.mean(self.length)*np.ones(nb_frame)
                self.alpha_mean = np.mean(self.alpha)*np.ones(nb_frame)
                self.beta_mean = np.mean(self.beta)*np.ones(nb_frame)
                self.gamma_mean = np.mean(self.gamma)*np.ones(nb_frame)

            nm_list = list()
            for ind_rm in range(0, len(rm)):
                nm = np.zeros((12, 3))

                nm_temp = vnop_array(rm[ind_rm]-self.rp,
                                     self.u, (self.rp-self.rd), self.w)

                nm_temp_mean = np.mean(nm_temp, axis=1)

                nm[0:3, :] = nm_temp_mean[0]*np.eye(3)
                nm[3:6, :] = (1+nm_temp_mean[1])*np.eye(3)
                nm[6:9, :] = -nm_temp_mean[1]*np.eye(3)
                nm[9:12, :] = nm_temp_mean[2]*np.eye(3)
                nm_list.append(nm)
            self.nm_list = nm_list
        else:
            # if the parameter are given it is already rigid
            self.length_mean = np.mean(
                segment_static.length_mean) * np.ones(nb_frame)
            self.alpha_mean = np.mean(
                segment_static.alpha_mean) * np.ones(nb_frame)
            self.beta_mean = np.mean(
                segment_static.beta_mean) * np.ones(nb_frame)
            self.gamma_mean = np.mean(
                segment_static.gamma_mean) * np.ones(nb_frame)

            self.nm_list = segment_static.nm_list

        self.Btype_prox = Btype_prox
        self.Btype_dist = Btype_dist
        self.Tprox = Q2T(self, Btype_prox, 'rp')
        self.Tdist = Q2T(self, Btype_dist, 'rd')

        if segment_static is None:
            if frame_prox is None:
                # Matrix allowing to go from the Tprox to the desired frame
                nb_frame = self.u.shape[1]
                X_eye = np.tile(np.array([1, 0, 0, 0])[
                                :, np.newaxis], (1, nb_frame))
                Y_eye = np.tile(np.array([0, 1, 0, 0])[
                                :, np.newaxis], (1, nb_frame))
                Z_eye = np.tile(np.array([0, 0, 1, 0])[
                                :, np.newaxis], (1, nb_frame))
                Or_eye = np.tile(np.array([0, 0, 0, 1])[
                                 :, np.newaxis], (1, nb_frame))
                self.corr_prox = HomogeneousMatrix(X_eye, Y_eye, Z_eye, Or_eye)
            else:

                corr_temp = self.Tprox.inv() * frame_prox
                corr_temp_mean = np.mean(corr_temp.T_homo, axis=2)
                self.corr_prox = HomogeneousMatrix.fromHomo(
                    np.repeat((corr_temp_mean[:, :, np.newaxis]), self.u.shape[1], axis=2))
        else:
            corr_temp_mean = np.mean(
                segment_static.corr_prox.T_homo, axis=2)
            self.corr_prox = HomogeneousMatrix.fromHomo(np.repeat(
                (corr_temp_mean[:, :, np.newaxis]), self.u.shape[1], axis=2))

        list_valid_name_inertia = ['platform', 'head', 'thorax', 'abdomen',
                                   'pelvis', 'arm', 'forearm', 'hand', 'thigh', 'tibia', 'foot']
        name_inertia_given = False
        for name in list_valid_name_inertia:
            if name in segment_name.lower():
                segment_name_inertia = name
                name_inertia_given = True

        if not name_inertia_given:
            segment_name_inertia = segment_name
            # Inertia properties
        if segment_name_inertia.lower() not in list_valid_name_inertia:
            segment_name_inertia = 'zero'
        # TODO : inertia properties need to be corrected relative to the side of the segment
        # in order to have correct position

        # If a specific inertia is given (to take into account zero inertia)
        if inertia == 'dumas':
            self.m, self.rCs, self.Is, Js_temp = dumas(
                weight, np.mean(self.length), sexe, segment_name_inertia)
        elif inertia == 'zero':
            self.m, self.rCs, self.Is, Js_temp = dumas(
                weight, np.mean(self.length), sexe, 'zero')
        # We add a dimension do be sure that tile multiply the matrix on the 3rd
        # dimension
        Js_temp = Js_temp[:, :, np.newaxis]

        self.Js = HomogeneousMatrix.fromHomo(
            np.tile(Js_temp, (1, 1, u.shape[1])))
        if phi_ext is None:
            self.phi_ext = HomogeneousMatrix.fromHomo(
                np.zeros((4, 4, u.shape[1])))
        else:
            self.phi_ext = phi_ext

        if nm_list is not None:
            print('marker_from_static')
            self.nm_list = nm_list

    @classmethod
    def fromSegment(cls, Segment, sexe='M', weight=0,
                    segment_static=None, rigid_parameter=False, inertia='dumas',
                    nm_list=None, frame_prox=None):

        return cls(Segment.u, Segment.rp, Segment.rd, Segment.w, Segment.rm,
                   Segment.Btype_prox, Segment.Btype_dist,
                   Segment.segment_name, Segment.rm_name, sexe, weight,
                   segment_static, rigid_parameter, inertia,
                   nm_list, frame_prox)

    def update(self):
        self.length = np.sqrt(np.sum((self.rp - self.rd)**2, axis=0))
        self.alpha = np.arccos(
            np.sum((self.rp - self.rd)*self.w, axis=0)/self.length)
        self.beta = np.arccos(np.sum(self.u*self.w, axis=0))
        self.gamma = np.arccos(
            np.sum(self.u*(self.rp-self.rd), axis=0)/self.length)
        self.Tprox = Q2T(self, self.Btype_prox, 'rp')
        self.Tdist = Q2T(self, self.Btype_dist, 'rd')

        return

    def get_distal_frame_glob(self):
        nb_frame = self.u.shape[1]
        X_glob = np.tile(np.array([1, 0, 0])[:, np.newaxis], (1, nb_frame))
        Y_glob = np.tile(np.array([0, 1, 0])[:, np.newaxis], (1, nb_frame))
        Z_glob = np.tile(np.array([0, 0, 1])[:, np.newaxis], (1, nb_frame))

        return HomogeneousMatrix(X_glob, Y_glob, Z_glob, self.rd)

    def get_proximal_frame_glob(self):
        nb_frame = self.u.shape[1]
        X_glob = np.tile(np.array([1, 0, 0])[:, np.newaxis], (1, nb_frame))
        Y_glob = np.tile(np.array([0, 1, 0])[:, np.newaxis], (1, nb_frame))
        Z_glob = np.tile(np.array([0, 0, 1])[:, np.newaxis], (1, nb_frame))

        return HomogeneousMatrix(X_glob, Y_glob, Z_glob, self.rp)

    def get_Q2T(self, Btype, origin_str):
        return Q2T(self, Btype, origin_str)
    # get_phim

    def get_phim(self):
        phim = np.zeros((len(self.rm)*3, 1, self.u.shape[1]))
        for ind_rm in range(0, len(self.rm)):
            phim[ind_rm*3:(ind_rm+1)*3, 0, :] = self.rm[ind_rm] - \
                np.dot(self.nm_list[ind_rm].T, self.Q)
        return phim
    # get_Km

    def get_Km(self):
        Km = np.zeros((3*len(self.nm_list), 12, 1))
        for ind_rm in range(0, len(self.nm_list)):
            Km[3*ind_rm:(ind_rm+1)*3, :, :] = - \
                self.nm_list[ind_rm].T[:, :, np.newaxis]
        return Km
    # get_Km

    def get_phir(self):
        phir = np.zeros((6, 1, self.u.shape[1]))
        phir[0, :, :] = np.sum(self.u**2, 0)-np.ones((self.u.shape[1]))
        phir[1, :, :] = np.sum(self.u*(self.rp-self.rd), 0) - self.length_mean * \
            np.cos(self.gamma_mean)
        phir[2, :, :] = np.sum(self.u*self.w, 0) - np.cos(self.beta_mean)
        phir[3, :, :] = np.sum((self.rp-self.rd)**2, 0) - \
            self.length_mean**2
        phir[4, :, :] = np.sum((self.rp-self.rd)*self.w, 0) - \
            self.length_mean*np.cos(self.alpha_mean)
        phir[5, :, :] = np.sum(self.w**2, 0)-np.ones(self.u.shape[1])

        return phir

    def get_Kr(self):
        # initialisation
        Kr = np.zeros((6, 12, self.u.shape[1]))

        Kr[0, 0:3, :] = 2*self.u

        Kr[1, 0:3, :] = self.rp-self.rd
        Kr[1, 3:6, :] = self.u
        Kr[1, 6:9, :] = -self.u

        Kr[2, 0:3, :] = self.w
        Kr[2, 9:12, :] = self.u

        Kr[3, 3:6, :] = 2*(self.rp-self.rd)
        Kr[3, 6:9, :] = -2*(self.rp-self.rd)

        Kr[4, 3:6, :] = self.w
        Kr[4, 6:9, :] = -self.w
        Kr[4, 9:12, :] = self.rp-self.rd

        Kr[5, 9:12, :] = 2*self.w

        return Kr

    def get_Weight_Matrix(self):
        W = np.zeros(
            (len(3*self.weight_rm), len(3*self.weight_rm), self.u.shape[1]))

        for ind_weight, value_weight in enumerate(self.weight_rm):
            W[3*(ind_weight):3*(ind_weight+1), 3*(ind_weight):3*(ind_weight+1),
              :] = value_weight*np.diag(np.ones(3))[:, :, np.newaxis]

        return W


def Q2T(self, Btype, origin_str):
    if Btype == 'Buv':
        B = Q2Buv(self.alpha, self.beta, self.gamma, self.length)
    elif Btype == 'Buw':
        B = Q2Buw(self.alpha, self.beta, self.gamma, self.length)
    elif Btype == 'Bwu':
        B = Q2Bwu(self.alpha, self.beta, self.gamma, self.length)

    if origin_str == 'rp':
        origin = self.rp
    elif origin_str == 'rd':
        origin = self.rd

    return Q2T_int(B, self.u, self.rp, self.rd, self.w, origin)


def Q2Buv(alpha, beta, gamma, length):
    # SYNOPSIS
    # T = Q2Tuv_array3(Q)
    #
    # INPUT
    # Q (i.e., generalized coordinates)
    #
    # OUTPUT
    # T (i.e., homogenous matrix)
    #
    # DESCRIPTION
    # Computation, for all frames (i.e., in 3rd dimension, cf. data structure
    # in user guide), of the homogenous matrix (T) from generalized coordinates
    # (Q) with origin at endpoint P and axis correspondence X = u,
    # Z = u x v / ||u x v||
    #
    # REFERENCE
    # R Dumas, L Cheze. 3D inverse dynamics in non-orthonormal segment
    # coordinate system. Medical & Biological Engineering & Computing 2007;
    # 45(3):315-22
    # R Dumas, T Robert, V Pomero, L Cheze. Joint and segment coordinate
    # systems revisited. Computer Methods in Biomechanics and Biomedical
    # Engineering. 2012;(15 Suppl 1):183-5

    nb_frame = alpha.shape[0]

    B = np.zeros((3, 3, nb_frame))
    B[0, 0, :] = np.ones((1, 1, nb_frame))
    B[0, 1, :] = (length*np.cos(gamma))
    B[0, 2, :] = np.cos(beta)
    B[1, 1, :] = (length*np.sin(gamma))

    btemp12 = ((np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma))
    B[1, 2, :] = btemp12
    b22temp = np.sqrt(1 - (np.cos(beta))**2
                      - ((np.cos(alpha) - np.cos(beta) *
                          np.cos(gamma)) / np.sin(gamma))**2
                      )
    B[2, 2, :] = b22temp

    return B


def Q2Buw(alpha, beta, gamma, length):
    # FUNCTION
    # Q2Tuw_array3.m
    # __________________________________________________________________________
    #
    # PURPOSE
    # Computation of homogenous matrix from generalized coordinates
    #
    # SYNOPSIS
    # T = Q2Tuw_array3(Q)
    #
    # INPUT
    # Q (i.e., generalized coordinates)
    #
    # OUTPUT
    # T (i.e., homogenous matrix)
    #
    # DESCRIPTION
    # Computation, for all frames (i.e., in 3rd dimension, cf. data structure
    # in user guide), of the homogenous matrix (T) from generalized coordinates
    # (Q) with origin at endpoint P and axis correspondence X = u,
    # Y = w x u / ||w x u||

    nb_frame = alpha.shape[0]
    B = np.zeros((3, 3, nb_frame))
    B[0, 0, :] = np.ones((1, 1, nb_frame))
    B[0, 1, :] = (length*np.cos(gamma))
    B[0, 2, :] = np.cos(beta)
    b11temp = np.sqrt(np.ones((1, nb_frame))-np.cos(gamma)**2
                      - ((np.cos(alpha)-np.cos(gamma)*np.cos(beta))/np.sin(beta))**2
                      )*length
    B[1, 1, :] = b11temp
    b21temp = length*(np.cos(alpha)-np.cos(gamma)*np.cos(beta))/np.sin(beta)
    B[2, 1, :] = b21temp
    B[2, 2, :] = np.sin(beta)
    return B


def Q2Bwu(alpha, beta, gamma, length):
    # FUNCTION
    # Q2Twu_array3.m
    # __________________________________________________________________________
    #
    # PURPOSE
    # Computation of homogenous matrix from generalized coordinates
    #
    # SYNOPSIS
    # T = Q2Twu_array3(Q)
    #
    # INPUT
    # Q (i.e., generalized coordinates)
    #
    # OUTPUT
    # T (i.e., homogenous matrix)
    #
    # DESCRIPTION
    # Computation, for all frames (i.e., in 3rd dimension, cf. data structure
    # in user guide), of the homogenous matrix (T) from generalized coordinates
    # (Q) with origin at endpoint D and axis correspondence Z = w,
    # Y = w x u / ||w x u||
    #
    # REFERENCE
    # R Dumas, T Robert, V Pomero, L Cheze. Joint and segment coordinate
    # systems revisited. Computer Methods in Biomechanics and Biomedical
    # Engineering. 2012;(15 Suppl 1):183-5
    nb_frame = alpha.shape[0]

    B = np.zeros((3, 3, nb_frame))
    B[0, 0, :] = (np.sin(beta))
    b01temp = length*(np.cos(gamma)-np.cos(alpha)*np.cos(beta))/np.sin(beta)
    B[0, 1, :] = b01temp
    b11temp = length*np.sqrt(np.ones(nb_frame)-np.cos(alpha)**2 -
                             ((np.cos(gamma)-np.cos(alpha) *
                               np.cos(beta))/np.sin(beta))**2
                             )
    B[1, 1, :] = b11temp
    B[2, 0, :] = (np.cos(beta))
    B[2, 1, :] = (length*np.cos(alpha))
    B[2, 2, :] = np.ones((1, 1, nb_frame))
    return B


def Q2T_int(B, u, rp, rd, w, Or):
    inv_B = np.zeros_like(B)
    for i in range(B.shape[-1]):
        inv_B[:, :, i] = np.linalg.inv(B[:, :, i])

    temp_Q = np.array([u, (rp-rd), w]).transpose((1, 0, 2))
    valid = np.einsum('mnr,ndr->mdr', temp_Q, inv_B)

    return HomogeneousMatrix.fromR_Or(valid, Or)
