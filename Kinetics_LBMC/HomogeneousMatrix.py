# -*- coding: utf-8 -*-
"""
Created on 18/10/2019

@author: Alexandre Naaim
"""
import numpy as np
from scipy import signal


class HomogeneousMatrix:
    """Creation of a homogeneous matrix using the X, Y, Z and Origin.

    References :
    Giovanni LEGNANI, Paolo RIGHETTINI, Bruno ZAPPA, Federico CASOLO
    A homogenous matrix approach to 3D kinematics and dynamics (1996)

    :param X:  X vector of the homogeneous matrix
    :type X: numpy.array shape= (3,1,nb_frame) or (4,1,nb_frame)
    :param Y: Y vector of the homogeneous matrix
    :type Y: numpy.array shape= (3,1,nb_frame) or (4,1,nb_frame)
    :param Z: Z vector of the homogeneous matrix
    :type Z: numpy.array shape= (3,1,nb_frame) or (4,1,nb_frame)
    :param Or: Description of parameter `Or`.
    :type Or: numpy.array shape= (3,1,nb_frame) or (4,1,nb_frame)
    :attr T_homo: Homogeneous matrx
    :type T_homo: numpy.array shape= (4,4,nb_frame)

    """

    def __init__(self, X, Y, Z, Or):
        """Creation of a homogeneous matrix using the X, Y, Z and Origin.

        References :
        Giovanni LEGNANI, Paolo RIGHETTINI, Bruno ZAPPA, Federico CASOLO
        A homogenous matrix approach to 3D kinematics and dynamics (1996)

        :param X:  X vector of the homogeneous matrix
        :type X: numpy.array shape= (3,1,nb_frame) or (4,1,nb_frame)
        :param Y: Y vector of the homogeneous matrix
        :type Y: numpy.array shape= (3,1,nb_frame) or (4,1,nb_frame)
        :param Z: Z vector of the homogeneous matrix
        :type Z: numpy.array shape= (3,1,nb_frame) or (4,1,nb_frame)
        :param Or: Description of parameter `Or`.
        :type Or: numpy.array shape= (3,1,nb_frame) or (4,1,nb_frame)
        :attr T_homo: Homogeneous matrx
        :type T_homo: numpy.array shape= (4,4,nb_frame)

        """
        nb_frame = X.shape[1]
        Homo = np.zeros((4, 4, nb_frame))
        if X.shape[0] == 3:
            Homo[0:3, 0, :] = X[0:3, :]
            Homo[0:3, 1, :] = Y[0:3, :]
            Homo[0:3, 2, :] = Z[0:3, :]
            Homo[0:3, 3, :] = Or[0:3, :]
            Homo[3, 3, :] = np.ones(nb_frame)
        elif X.shape[0] == 4:
            Homo[:, 0, :] = X
            Homo[:, 1, :] = Y
            Homo[:, 2, :] = Z
            Homo[:, 3, :] = Or

        self.T_homo = Homo

    @classmethod
    def fromR_Or(cls, R, Or):
        """Create a homogeneous matrix using a rotation matrix R and an origin Or.

        :param cls: Description of parameter `cls`.
        :type cls: type
        :param R: Rotation matrix
        :type R: numpy.array shape =(3,3,nb_frame)
        :param Or: Origin of the homogeneous matrix.
        :type Or: np.array.shape = (3,1,nb_frame)
        :return: Homogeneous Matrix
        :rtype: HomogeneousMatrix

        """
        X = R[0:3, 0, :]
        Y = R[0:3, 1, :]
        Z = R[0:3, 2, :]
        return cls(X, Y, Z, Or)

    @classmethod
    def fromHomo(cls, Homo):
        """Create a homogeneous matrix using a homogenous matrix.

        :param cls: Description of parameter `cls`.
        :type cls: type
        :param Homo: Homogeneous matrix
        :type Homo: numpy.array shape = (4,4,nb_frame)
        :return: Homogeneous matrix
        :rtype: HomogeneousMatrix

        """
        X = Homo[:, 0, :]
        Y = Homo[:, 1, :]
        Z = Homo[:, 2, :]
        Or = Homo[:, 3, :]
        return cls(X, Y, Z, Or)

    def vel_acc_Mat(self, frq_point, fq_cutoff):
        """Calculate the H and W homogeneous matrix.
        TODO : Explain what are those two beautiful matrix

        W = M'inv(M)
        H = M''inv(M)

        G Legnani, F Casolo, P Righettini, B Zappa, A homogeneous matrix approach
        to 3D kinematics and dynamics - I. Theory. Mechanisms and Machine Theory
        1996;31(5):573–87

        :param frq_point:  Point frequency
        :type frq_point: double
        :param fq_cutoff: Cut-off frequency.
        :type fq_cutoff: double
        :return: W and H homogeneous matrix
        :rtype: list of HomogeneousMatrix

        """
        dt = 1 / frq_point
        grad = np.gradient(self.T_homo, dt, axis=2)
        grad_2 = np.gradient(grad, dt, axis=2)

        W = np.einsum("mnr,ndr->mdr", grad, self.inv().T_homo)
        H = np.einsum("mnr,ndr->mdr", grad_2, self.inv().T_homo)

        b, a = signal.butter(4, fq_cutoff / (0.5 * frq_point), btype="lowpass")
        W[0:3, :, :] = signal.filtfilt(b, a, W[0:3, :, :], axis=2)
        H[0:3, :, :] = signal.filtfilt(b, a, H[0:3, :, :], axis=2)

        return HomogeneousMatrix.fromHomo(W), HomogeneousMatrix.fromHomo(H)

    def inv(self):
        """Inversion of the homogeneous matrix (only for position homogeneous matrix).

        :return: Inverse homogenous matrix
        :rtype: HomogeneousMatrix

        """
        # inverse of a Homogenous matrice is [R.T,-R.T*Or]
        R_T = self.T_homo[0:3, 0:3, :].transpose(1, 0, 2)
        Or_init = self.T_homo[0:3, 3, :]
        Or_inv_2 = np.einsum("mnr,nr->mr", -R_T, Or_init)
        return HomogeneousMatrix.fromR_Or(R_T, Or_inv_2)

    def transpose(self):
        """Transpose the homogeneous matrix.

        :return: Transpose of the homogenous matrix
        :rtype: HomogeneousMatrix

        """
        transpose_homo = self.T_homo.transpose(1, 0, 2)
        return HomogeneousMatrix.fromHomo(transpose_homo)

    def __mul__(self, other):
        """ """
        return HomogeneousMatrix.fromHomo(np.einsum("mnr,ndr->mdr", self.T_homo, other.T_homo))

    def __add__(self, other):
        return HomogeneousMatrix.fromHomo(self.T_homo + other.T_homo)

    def __sub__(self, other):
        return HomogeneousMatrix.fromHomo(self.T_homo - other.T_homo)
