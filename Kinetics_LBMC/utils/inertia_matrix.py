# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict


def dumas(weight, length_segment, sexe, segment_name):
    """Integration of the inertia parameter extracted from Dumas et al. 2018.

    Estimation of the Body Segment Inertial Parameters for the Rigid Body
    Biomechanical Models Used in Motion Analysis
    Raphaël Dumas and Janis Wojtusch
    These inertia parameter are valid only if the program is computed respecting
    ISB convention

    :param type weight: Weight of the subject in kg
    :param type length_segment: Lenght of the segment (unit should be the same
    as in the rest of the program)
    :param type sexe: Sexe of the subject
    :param type segment_name: name of the segment (can be platform, foot,
    tibia, thigh, pelvis)
    :return: List with the mass of the segement(ms), the position of the center
    of mass respect to the segment reference frame (rCs)
    the inertia matrix with respect to the center of mass Is, and the Homogenous
     matrix of pseudo-inertia expressed in SCS (Js)
    :rtype: list

    """
    def nested_dict(): return defaultdict(nested_dict)
    inertia = nested_dict()

    inertia['mass_perc']['F']['platform'] = 0
    inertia['mass_perc']['F']['zero'] = 0

    inertia['mass_perc']['F']['head'] = 6.7
    inertia['mass_perc']['F']['thorax'] = 26.3
    inertia['mass_perc']['F']['abdomen'] = 4.1
    inertia['mass_perc']['F']['pelvis'] = 14.7
    inertia['mass_perc']['F']['arm'] = 2.3
    inertia['mass_perc']['F']['forearm'] = 1.4
    inertia['mass_perc']['F']['hand'] = 0.5
    inertia['mass_perc']['F']['thigh'] = 14.6
    inertia['mass_perc']['F']['tibia'] = 4.5
    inertia['mass_perc']['F']['foot'] = 1

    inertia['perc_length']['F']['platform'] = np.array([0, 0, 0])
    inertia['perc_length']['F']['zero'] = np.array([0, 0, 0])

    inertia['perc_length']['F']['head'] = np.array([0.8, 55.9, -0.1])/100
    inertia['perc_length']['F']['thorax'] = np.array([1.5, -54.2, 0.1])/100
    inertia['perc_length']['F']['abdomen'] = np.array([21.9, -41, 0.3])/100
    inertia['perc_length']['F']['pelvis'] = np.array([-7.2, -22.8, 0.2])/100
    inertia['perc_length']['F']['arm'] = np.array(
        [-5.5, -50.0, -3.3])/100
    inertia['perc_length']['F']['forearm'] = np.array([2.1, -41.1, 1.9])/100
    inertia['perc_length']['F']['hand'] = np.array([7.7, -76.8, 4.8])/100
    inertia['perc_length']['F']['thigh'] = np.array([-7.7, -37.7, 0.8])/100
    inertia['perc_length']['F']['tibia'] = np.array([-4.9,  -40.4, 3.1])/100
    inertia['perc_length']['F']['foot'] = np.array([38.2, -30.9, 5.5])/100

    # 10000 = 100^2 because it is the percentage that is at the power of 2
    inertia['J']['F']['zero'] = np.array([[0, 0,  0],
                                          [0, 0, 0],
                                          [0, 0,  0]])**2/10000
    inertia['J']['F']['platform'] = np.array([[0, 0,  0],
                                              [0, 0, 0],
                                              [0, 0,  0]])**2/10000

    inertia['J']['F']['head'] = np.array([[30, -5,  -1],
                                          [-5, 24,  -0],
                                          [-1,  -0, 31]])**2/10000
    inertia['J']['F']['thorax'] = np.array([[38, -12, -3],
                                            [-12, 32,  -1],
                                            [-3,  -1, 34]])**2/10000
    inertia['J']['F']['abdomen'] = np.array([[65, -25, -3],
                                             [-25, 78,  -5],
                                             [-3,  -5, 52]])**2/10000
    inertia['J']['F']['arm'] = np.array([[30, -3, -5],
                                         [-3, 15,  -3],
                                         [-5,  -3, 30]])**2/10000
    inertia['J']['F']['forearm'] = np.array([[27, -10, -3],
                                             [-10, 14,  -13],
                                             [-3,  -13, 25]])**2/10000
    inertia['J']['F']['hand'] = np.array([[64, -29, -23],
                                          [-29, 43,  -28],
                                          [-23,  -28, 59]])**2/10000

    inertia['J']['F']['pelvis'] = np.array([[95, -35, -3],
                                            [-35, 105, -2],
                                            [-3, -2, 82]])**2/10000
    # TODO What sign for the value out of the diagonal :
    inertia['J']['F']['thigh'] = np.array([[31, -7,  2],
                                           [-7, 19, -7],
                                           [2,  -7, 32]])**2/10000
    inertia['J']['F']['tibia'] = np.array([[28, 2, 1],
                                           [2, 10, 6],
                                           [1, 6, 28]])**2/10000
    inertia['J']['F']['foot'] = np.array([[24, -15,  9],
                                          [-15, 50, -5],
                                          [9,   -5, 50]])**2/10000

    inertia['mass_perc']['M']['platform'] = 0
    inertia['mass_perc']['M']['zero'] = 0

    inertia['mass_perc']['M']['head'] = 6.7
    inertia['mass_perc']['M']['thorax'] = 30.4
    inertia['mass_perc']['M']['abdomen'] = 2.9
    inertia['mass_perc']['M']['pelvis'] = 14.2
    inertia['mass_perc']['M']['arm'] = 2.4
    inertia['mass_perc']['M']['forearm'] = 1.7
    inertia['mass_perc']['M']['hand'] = 0.6
    inertia['mass_perc']['M']['thigh'] = 12.3
    inertia['mass_perc']['M']['tibia'] = 4.8
    inertia['mass_perc']['M']['foot'] = 1.2

    inertia['perc_length']['M']['platform'] = np.array([0, 0, 0])
    inertia['perc_length']['M']['zero'] = np.array([0, 0, 0])
    inertia['perc_length']['M']['head'] = np.array([2.0, 53.4, 0.1])/100
    inertia['perc_length']['M']['thorax'] = np.array([0.0, -55.5, -0.4])/100
    inertia['perc_length']['M']['abdomen'] = np.array([17.6, -36.1, -3.3])/100
    inertia['perc_length']['M']['pelvis'] = np.array([-0.2, -28.2, -0.6])/100
    inertia['perc_length']['M']['arm'] = np.array([1.8, -48.2, -3.1])/100
    inertia['perc_length']['M']['forearm'] = np.array([-1.3, -41.7, 1.1])/100
    inertia['perc_length']['M']['hand'] = np.array([8.2, -83.9, 7.5])/100

    inertia['perc_length']['M']['thigh'] = np.array([-4.1, -42.9, 3.3])/100
    inertia['perc_length']['M']['tibia'] = np.array([-4.8,  -41, 0.7])/100
    inertia['perc_length']['M']['foot'] = np.array([50.2, -19.9, 3.4])/100

    inertia['J']['M']['platform'] = np.array([[0, 0,  0],
                                              [0, 0, 0],
                                              [0, 0,  0]])**2/10000
    inertia['J']['M']['zero'] = np.array([[0, 0,  0],
                                          [0, 0, 0],
                                          [0, 0,  0]])**2/10000

    inertia['J']['M']['head'] = np.array([[28, -7, -2],
                                          [-7, 21,  -3],
                                          [-2,  -3, 30]])**2/10000
    inertia['J']['M']['thorax'] = np.array([[42, -11, -1],
                                            [-11, 33,  -3],
                                            [-1,  -3, 36]])**2/10000
    inertia['J']['M']['abdomen'] = np.array([[54, -11, -6],
                                             [-11, 66,  -5],
                                             [-6,  -5, 40]])**2/10000
    inertia['J']['M']['pelvis'] = np.array([[102, -25, -12],
                                            [-25, 106,  -8],
                                            [-12,  -8,  96]])**2/10000
    inertia['J']['M']['arm'] = np.array([[29, -5, -3],
                                         [-5, 13,  -13],
                                         [-3,  -13, 30]])**2/10000
    inertia['J']['M']['forearm'] = np.array([[28, -8, -1],
                                             [-8, 11,  -2],
                                             [-1,  -2, 28]])**2/10000
    inertia['J']['M']['hand'] = np.array([[61, -22, -15],
                                          [-22, 38,  -20],
                                          [-15,  -20, 56]])**2/10000

    # TODO What sign for the value out of the diagonal
    inertia['J']['M']['thigh'] = np.array([[29,  7, -2],
                                           [7,  15, -7],
                                           [-2, -7, 30]])**2/10000
    inertia['J']['M']['tibia'] = np.array([[28, -4, -2],
                                           [-4, 10,  4],
                                           [-2,  4, 28]])**2/10000
    inertia['J']['M']['foot'] = np.array([[22, 17, -11],
                                          [17, 49,   0],
                                          [-11, 0,  48]])**2/10000

    ms = inertia['mass_perc'][sexe][segment_name.lower()]*weight/100
    rCs = length_segment * inertia['perc_length'][sexe][segment_name.lower()]
    Is = ms*length_segment**2 * inertia['J'][sexe][segment_name.lower()]

    # Homogenous matrix of pseudo-inertia expressed in SCS (Js)
    # The homogenous formulation is not the same for the diagonal
    # When the matrix is transported to the proximal point the I is not changed
    Js = np.zeros((4, 4))

    Js[0: 3, 0: 3] = Is.trace()/2*np.eye(3)-Is

    Js[3, 0: 3] = ms*rCs
    Js[0: 3, 3] = ms*rCs
    Js[3, 3] = ms
    return(ms, rCs, Is, Js)


def zeros_inertia(weight, length_segment, sexe, segment_name):
    """Integration of the inertia parameter empty.

    :param type weight: Weight of the subject in kg
    :param type length_segment: Lenght of the segment (unit should be the same
    as in the rest of the program)
    :param type sexe: Sexe of the subject
    :param type segment_name: name of the segment (can be platform, foot,
    tibia, thigh, pelvis)
    :return: List with the mass of the segement(ms), the position of the center
    of mass respect to the segment reference frame (rCs)
    the inertia matrix with respect to the center of mass Is, and the Homogenous
     matrix of pseudo-inertia expressed in SCS (Js)
    :rtype: list

    """
    def nested_dict(): return defaultdict(nested_dict)
    inertia = nested_dict()

    inertia['mass_perc']['F']['platform'] = 0
    inertia['mass_perc']['F']['foot'] = 0
    inertia['mass_perc']['F']['tibia'] = 0
    inertia['mass_perc']['F']['thigh'] = 0
    inertia['mass_perc']['F']['pelvis'] = 0

    inertia['perc_length']['F']['platform'] = np.array([0, 0, 0])
    inertia['perc_length']['F']['foot'] = np.array([38.2, -30.9, 5.5])/100
    inertia['perc_length']['F']['tibia'] = np.array([-4.9,  -40.4, 3.1])/100
    inertia['perc_length']['F']['thigh'] = np.array([-7.7, -37.7, 0.8])/100
    inertia['perc_length']['F']['pelvis'] = np.array([-7.7, -37.7, 0.8])/100

    inertia['J']['F']['foot'] = np.array([[0, 0,  0],
                                          [0, 0, 0],
                                          [0, 0,  0]])**2/10000
    inertia['J']['F']['foot'] = np.array([[0, 0,  0],
                                          [0, 0, 0],
                                          [0, 0,  0]])**2/10000
    inertia['J']['F']['tibia'] = np.array([[0, 0,  0],
                                           [0, 0, 0],
                                           [0, 0,  0]])**2/10000
    inertia['J']['F']['thigh'] = np.array([[0, 0,  0],
                                           [0, 0, 0],
                                           [0, 0,  0]])**2/10000
    inertia['J']['F']['pelvis'] = np.array([[0, 0,  0],
                                            [0, 0, 0],
                                            [0, 0,  0]])**2/10000

    inertia['mass_perc']['M']['platform'] = 0
    inertia['mass_perc']['M']['foot'] = 0
    inertia['mass_perc']['M']['tibia'] = 0
    inertia['mass_perc']['M']['thigh'] = 0
    inertia['mass_perc']['M']['pelvis'] = 0

    inertia['perc_length']['M']['platform'] = np.array([0, 0, 0])
    inertia['perc_length']['M']['foot'] = np.array([50.2, -19.9, 3.4])/100
    inertia['perc_length']['M']['tibia'] = np.array([-4.8,  -41, 0.7])/100
    inertia['perc_length']['M']['thigh'] = np.array([-4.1, -42.9, 3.3])/100
    inertia['perc_length']['M']['pelvis'] = np.array([-0.2, -28.2, 0.6])/100

    inertia['J']['M']['platform'] = np.array([[0, 0,  0],
                                              [0, 0, 0],
                                              [0, 0,  0]])**2/10000
    inertia['J']['M']['foot'] = np.array([[0, 0,  0],
                                          [0, 0, 0],
                                          [0, 0,  0]])**2/10000
    inertia['J']['M']['tibia'] = np.array([[0, 0,  0],
                                           [0, 0, 0],
                                           [0, 0,  0]])**2/10000
    inertia['J']['M']['thigh'] = np.array([[0, 0,  0],
                                           [0, 0, 0],
                                           [0, 0,  0]])**2/10000
    inertia['J']['M']['pelvis'] = np.array([[0, 0,  0],
                                            [0, 0, 0],
                                            [0, 0,  0]])**2/10000

    ms = inertia['mass_perc'][sexe][segment_name.lower()]*weight/100/9.81
    rCs = length_segment * inertia['perc_length'][sexe][segment_name.lower()]
    Is = ms*length_segment**2 * inertia['J'][sexe][segment_name.lower()]

    # Homogenous matrix of pseudo-inertia expressed in SCS (Js)
    Js = np.zeros((4, 4))
    Js[0: 3, 0: 3] = Is.trace()/2*np.eye(3)-Is
    Js[3, 0: 3] = ms*rCs
    Js[0: 3, 3] = ms*rCs
    Js[3, 3] = ms

    return(ms, rCs, Is, Js)
