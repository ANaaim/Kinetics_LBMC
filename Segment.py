from vnop_array import vnop_array as vnop_array
import inertia_matrix
from HomogeneousMatrix import HomogeneousMatrix
import numpy as np
# -*- coding: utf-8 -*-
"""Segment class used to define anatomical segment based on natural coordinate.

Created on Wed Feb 20 13:21:05 2019

@author: AdminXPS
"""


class Segment:

    def __init__(self, u, rp, rd, w, rm,
                 Btype_prox, Btype_dist,
                 segment_name, sexe='M', weight=0,
                 segment_static=None, rigid_parameter=False, inertia='dumas',
                 nm_list=None):

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

        # TODO : Create a constructor where these parameters are given
        if segment_static is None:
            self.length = np.sqrt(np.sum((rp - rd)**2, axis=0))
            self.alpha = np.arccos(np.sum((rp - rd)*w, axis=0)/self.length)
            self.beta = np.arccos(np.sum(u*w, axis=0))
            self.gamma = np.arccos(np.sum(u*(rp-rd), axis=0)/self.length)
            if rigid_parameter:
                self.length = np.mean(self.length)*np.ones(nb_frame)
                self.alpha = np.mean(self.alpha)*np.ones(nb_frame)
                self.beta = np.mean(self.beta)*np.ones(nb_frame)
                self.gamma = np.mean(self.gamma)*np.ones(nb_frame)

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
            self.length = np.mean(segment_static.length) * np.ones(nb_frame)
            self.alpha = np.mean(segment_static.alpha) * np.ones(nb_frame)
            self.beta = np.mean(segment_static.beta) * np.ones(nb_frame)
            self.gamma = np.mean(segment_static.gamma) * np.ones(nb_frame)

            self.nm_list = segment_static.nm_list

        self.Btype_prox = Btype_prox
        self.Btype_dist = Btype_dist
        self.Tprox = Q2T(self, Btype_prox, 'rp')
        self.Tdist = Q2T(self, Btype_dist, 'rd')

        # Inertia properties
        if segment_name.lower() not in ['plateform', 'foot', 'tibia', 'tigh', 'pelvis']:
            segment_name = 'zero'

        # If a specific inertia is given (to take into account zero inertia)
        if inertia is 'dumas':
            self.m, self.rCs, self.Is, Js_temp = inertia_matrix.dumas(
                weight, np.mean(self.length), sexe, segment_name)
        elif inertia is 'zero':
            self.m, self.rCs, self.Is, Js_temp = inertia_matrix.dumas(
                weight, np.mean(self.length), sexe, 'zero')
        # We add a dimension do be sure that tile multiply the matrix on the 3rd
        # dimension
        Js_temp = Js_temp[:, :, np.newaxis]
        self.Js = HomogeneousMatrix.fromHomo(
            np.tile(Js_temp, (1, 1, u.shape[1])))

        if nm_list is not None:
            print('marker_from_static')
            self.nm_list = nm_list

    @classmethod
    def fromSegment(cls, Segment, sexe='M', weight=0,
                    segment_static=None, rigid_parameter=False, inertia='dumas',
                    nm_list=None):

        return cls(Segment.u, Segment.rp, Segment.rd, Segment.w, Segment.rm,
                   Segment.Btype_prox, Segment.Btype_dist,
                   Segment.segment_name, sexe, weight,
                   segment_static, rigid_parameter, inertia,
                   nm_list)

    def update(self):
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
        phir[1, :, :] = np.sum(self.u*(self.rp-self.rd), 0) - self.length * \
            np.cos(self.gamma)
        phir[2, :, :] = np.sum(self.u*self.w, 0) - np.cos(self.beta)
        phir[3, :, :] = np.sum((self.rp-self.rd)**2, 0) - \
            self.length**2
        phir[4, :, :] = np.sum((self.rp-self.rd)*self.w, 0) - \
            self.length*np.cos(self.alpha)
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
