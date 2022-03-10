# -*- coding: utf-8 -*-
import numpy as np
from .HomogeneousMatrix import HomogeneousMatrix
from .utils.norm_vector import norm_vector as norm_vector


class Joint:
    """A class used to represente a antomical joint.

    :param segment_dist: Generally the distal Segment of the joint.
    :type segment_dist: Segment
    :param seg_phi_dist: Action homogenous matrix at the distal point of the Segment
    (Legnagni et al 1996)
    :type seg_phi_dist: HomogeneousMatrix
    :param frq_acq: point frequency
    :type frq_acq: double
    :param frq_cp: Cut off frequency for the filetering after derivation
    :type frq_cp: double
    :attr phi_prox_origin: Action homogenous matrix at the proximal point of the
    Segment
    :type phi_prox_origin: HomogeneousMatrix

    """

    def __init__(self, segment_dist, seg_phi_dist, frq_acq, frq_cp, gravity_direction, unit_point):
        """Construct the Joint object.

        :param segment_dist: Generally the distal Segment of the joint.
        :type segment_dist: Segment
        :param seg_phi_dist:Action homogenous matrix at the distal point of the Segment
        :type seg_phi_dist: type
        :param frq_acq: point frequency
        :type frq_acq: double
        :param frq_cp: Cut off frequency for the filetering after derivation
        :type frq_cp: double
        :return: Joint
        :rtype: Joint

        """
        # seg_phi_dist should be expressed at the origin
        nb_frame = segment_dist.Tprox.T_homo.shape[2]
        # Gravity
        if unit_point == 'mm':
            norm_gravity = 9810
        elif unit_point == 'm':
            norm_gravity = 9.81
        else:
            print('The unit '+unit_point +
                  ' cannot be considered for dynamic calculation. It should be m or mm.')

        Hg = np.zeros((4, 4, nb_frame))
        Hg[abs(gravity_direction), 3, :] = (
            gravity_direction/abs(gravity_direction))*norm_gravity
        Hg = HomogeneousMatrix.fromHomo(Hg)

        # Projection of the J from the proximal position to the origin
        J_temp = segment_dist.Tprox * \
            (segment_dist.Js * segment_dist.Tprox.transpose())
        # Calculaiton of W and H at the origine
        W_segment, H_segment = segment_dist.Tprox.vel_acc_Mat(frq_acq, frq_cp)

        acc_rel = H_segment-Hg
        self.phi_prox_origin = seg_phi_dist + acc_rel*J_temp-J_temp*acc_rel.transpose()

    def get_force_moment(self, frame, base):
        # frame is a homogeneous matrix in which the force shoud be expressed
        # is the frame in which the force should be expressed
        phi_projected = frame.inv()*(self.phi_prox_origin*frame.inv().transpose())

        F, M = extraction_force_moment_from_phi(phi_projected)

        R = base.inv().T_homo[0:3, 0:3, :]
        M = np.einsum('mnr,ndr->mdr', R, M)
        F = np.einsum('mnr,ndr->mdr', R, F)

        return F, M

    def projection_JCS(self, joint_center, segment_prox, segment_dist, rotation_seq):
        # Transport of the phi_origin to the joint_center in the world frame
        nb_frame = self.phi_prox_origin.T_homo.shape[2]
        X_glob = np.tile(np.array([1, 0, 0])[:, np.newaxis], (1, nb_frame))
        Y_glob = np.tile(np.array([0, 1, 0])[:, np.newaxis], (1, nb_frame))
        Z_glob = np.tile(np.array([0, 0, 1])[:, np.newaxis], (1, nb_frame))
        # print(joint_center.shape)

        joint_center_mathomo = HomogeneousMatrix(
            X_glob, Y_glob, Z_glob, joint_center)

        phi_projected = joint_center_mathomo.inv() * \
            (self.phi_prox_origin*joint_center_mathomo.inv().transpose())

        # extraction of F and M from phi_projected
        F, M = extraction_force_moment_from_phi(phi_projected)

        # Extraction of the different frame used for the rotation
        frame_prox = segment_prox.Tdist.T_homo
        frame_dis = segment_dist.Tprox.T_homo

        # Calculation of the different axis of rotation
        if rotation_seq.lower() == 'zyx':
            rot_z = frame_prox[:3, 2, :]
            rot_x = frame_dis[:3, 0, :]
            rot_y = norm_vector(
                np.cross(rot_z, rot_x, axisa=0, axisb=0, axisc=0))
        elif rotation_seq.lower() == 'zxy':
            rot_z = frame_prox[:3, 2, :]
            rot_y = frame_dis[:3, 1, :]
            rot_x = norm_vector(
                np.cross(rot_y, rot_z, axisa=0, axisb=0, axisc=0))
        # projection of F and M on the different axis.
        M_JCS = np.zeros((3, 1, nb_frame))
        F_JCS = np.zeros((3, 1, nb_frame))

        M_JCS[0, 0, :] = np.sum(rot_x*M[:, 0, :], axis=0)
        M_JCS[1, 0, :] = np.sum(rot_y*M[:, 0, :], axis=0)
        M_JCS[2, 0, :] = np.sum(rot_z*M[:, 0, :], axis=0)

        F_JCS[0, 0, :] = np.sum(rot_x*F[:, 0, :], axis=0)
        F_JCS[1, 0, :] = np.sum(rot_y*F[:, 0, :], axis=0)
        F_JCS[2, 0, :] = np.sum(rot_z*F[:, 0, :], axis=0)

        return F_JCS, M_JCS


def extraction_force_moment_from_phi(phi):
    nb_frame = phi.T_homo.shape[2]
    F = np.zeros((3, 1, nb_frame))
    M = np.zeros((3, 1, nb_frame))
    F[0, 0, :] = (phi.T_homo[0, 3, :]-phi.T_homo[3, 0, :])/2
    F[1, 0, :] = (phi.T_homo[1, 3, :]-phi.T_homo[3, 1, :])/2
    F[2, 0, :] = (phi.T_homo[2, 3, :]-phi.T_homo[3, 2, :])/2

    M[0, 0, :] = (-phi.T_homo[1, 2, :]+phi.T_homo[2, 1, :])/2
    M[1, 0, :] = (phi.T_homo[0, 2, :]-phi.T_homo[2, 0, :])/2
    M[2, 0, :] = (-phi.T_homo[0, 1, :]+phi.T_homo[1, 0, :])/2

    return F, M
