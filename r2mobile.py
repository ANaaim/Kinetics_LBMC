# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:09:18 2019

@author: AdminXPS
"""
import numpy as np


def zyx(rotation_matrix):
    """Computation of Euler angles from rotation matrix (with ZYX mobile
    sequence for joint kinematics)

    Computation, for all frames (i.e., in 3rd dimension, cf. data structure
    in user guide), of the Euler angles (tetha1, tetha2, tetha3) from the
    rotation matrix (R) using a sequence of mobile axes ZYX

    References :
    GK Cole, BM Nigg, JL Ronsky MR Yeadon. Application of the joint
    coordinate system to three-dimensional joint attitude and movement
    representation: a standardisation proposal. Journal of Biomechanical
    Engineering 1993; 115(4): 344-349
    R Dumas, T Robert, V Pomero, L Cheze. Joint and segment coordinate
    systems revisited. Computer Methods in Biomechanics and Biomedical
    Engineering 2012;15(Suppl 1):183-5

    :param rotation_matrix: Description of parameter `rotation_matrix`.
    :type rotation_matrix: numpy array (shape can be (3,3,nb_frame) or
    (4,4,nb_frame))
    :return:Joint_Euler_Angles (i.e., tetha1, tetha2, tetha3, in line)
    :rtype: numpy.array shape (3,nb_frame)

    """
    # Tetha1 (about Z proximal SCS axis):
    # e.g., flexion-extension at the ankle
    rot_z = np.arctan2(rotation_matrix[1, 0, :],
                       rotation_matrix[0, 0, :])
    # Tetha2 (about Y floating axis):
    # e.g., abduction-adduction at the ankle
    rot_y = np.arcsin(-rotation_matrix[2, 0, :])
    # Tetha3 (about X distal SCS axis):
    # e.g., internal-external rotation at the ankle
    rot_x = np.arctan2(rotation_matrix[2, 1, :],
                       rotation_matrix[2, 2, :])
    return np.array([rot_z, rot_y, rot_x])


def zxy(rotation_matrix):
    """Computation of Euler angles from rotation matrix (with ZXY mobile
    sequence for joint kinematics)

    Computation, for all frames (i.e., in 3rd dimension, cf. data structure
    in user guide), of the Euler angles (tetha1, tetha2, tetha3) from the
    rotation matrix (R) using a sequence of mobile axes ZXY

    References :
    G Wu, S Siegler, P Allard, C Kirtley, A Leardini, D Rosenbaum, M Whittle,
    DD D'Lima, L Cristofolini, H Witte, O Schmid, I Stokes. ISB
    recommendation on definitions of joint coordinate system of various
    joints for the reporting of human joint motion - Part I: ankle, hip, and
    spine. Journal of Biomechanics 2002;35(4):543-8.
    G Wu, FC van der Helm, HE Veeger, M Makhsous, P Van Roy, C Anglin,
    J Nagels, AR Karduna, K McQuade, X Wang, FW Werner, B Buchholz. ISB
    recommendation on definitions of joint coordinate systems of various
    joints for the reporting of human joint motion - Part II: shoulder,
    elbow, wrist and hand. Journal of Biomechanics 2005;38(5):981-92.

    :param rotation_matrix: Description of parameter `rotation_matrix`.
    :type rotation_matrix: numpy array (shape can be (3,3,nb_frame) or
    (4,4,nb_frame))
    :return:Joint_Euler_Angles (i.e., tetha1, tetha2, tetha3, in line)
    :rtype: numpy.array shape (3,nb_frame)

    """
    # Tetha1: Flexion-Extension (about Z proximal SCS axis)
    rot_z = np.arctan2(-rotation_matrix[0, 1, :],
                       rotation_matrix[1, 1, :])
    # Tetha2: Abduction-Adduction (about X floating axis)
    rot_x = np.arcsin(rotation_matrix[2, 1, :])
    # Tetha3: Internal-External Rotation (about Y distal SCS axis)
    rot_y = np.arctan2(-rotation_matrix[2, 0, :],
                       rotation_matrix[2, 2, :])
    return np.array([rot_z, rot_x, rot_y])


def xzy(rotation_matrix):
    """Computation of Euler angles from rotation matrix (with XZY mobile
    sequence for joint kinematics)

    Computation, for all frames (i.e., in 3rd dimension, cf. data structure
    in user guide), of the Euler angles (tetha1, tetha2, tetha3) from the
    rotation matrix (R) using a sequence of mobile axes XZY

    References :
    M Senk, L Cheze. Rotation sequence as an important factor in shoulder
    kinematics. Clinical Biomechanics 2006;21(S1):S3-8
    A Bonnefoy-Mazure, J Slawinski, A Riquet, JM Lévèque, C Miller,  L Cheze.
    Rotation sequence is an important factor in shoulder kinematics.
    Application to the elite players' flat serves. Journal of Biomechanics
    2010;43(10):2022-5

    :param rotation_matrix: Description of parameter `rotation_matrix`.
    :type rotation_matrix: numpy array (shape can be (3,3,nb_frame) or
    (4,4,nb_frame))
    :return:Joint_Euler_Angles (i.e., tetha1, tetha2, tetha3, in line)
    :rtype: numpy.array shape (3,nb_frame)

    """
    # Tetha1 (about X proximal SCS axis):
    # e.g., abduction-adduction at the shoulder
    rot_x = np.arctan2(rotation_matrix[2, 1, :],
                       rotation_matrix[1, 1, :])
    # Tetha2 (about Z floating axis):
    # e.g., flexion-extension at the shoulder
    rot_z = np.arcsin(-rotation_matrix[0, 1, :])
    # Tetha3 (about Y distal SCS axis):
    # e.g., internal-external rotation at the shoulder
    rot_y = np.arctan2(rotation_matrix[0, 2, :],
                       rotation_matrix[0, 0, :])
    return np.array([rot_x, rot_z, rot_y])
