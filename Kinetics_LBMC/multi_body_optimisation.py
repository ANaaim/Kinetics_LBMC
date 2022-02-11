# -*- coding: utf-8 -*-
"""
Created on 18/10/2019

@author: Alexandre Naaim
"""
import numpy as np


def multi_body_optimisation(full_segment, full_model, max_step=50):

    nb_frame = full_segment[0].u.shape[1]
    nb_segment = len(full_segment)
    nb_marker = 0
    for segment in full_segment:
        nb_marker = nb_marker + len(segment.rm)

    nb_constraint = 0
    for model in full_model:
        nb_constraint += model.nb_constraint

    # Initialisation
    dKlambdardQ = np.zeros((12*nb_segment, 12*nb_segment, nb_frame))
    lambda_r = np.zeros((nb_segment*6, 1, nb_frame))
    lambda_k = np.zeros((nb_constraint, 1, nb_frame))

    # Construction des éléments constants
    Km = np.zeros((3*nb_marker, nb_segment*12, nb_frame))
    ind_marker = 0
    for ind_segment, segment in enumerate(full_segment):
        Km[ind_marker*3:(ind_marker+len(segment.rm))*3,
           ind_segment*12:(ind_segment+1)*12, :] = \
            np.tile(segment.get_Km(), (1, 1, nb_frame))
        ind_marker = ind_marker + len(segment.rm)

    Full_KmT = np.einsum('mnr,ndr->mdr', np.transpose(Km, (1, 0, 2)), Km)
    error = 1
    step = 0

    while error > 10**-12 and step < max_step:
        step = step + 1
        print(step)
        # initialisation
        ind_marker = 0
        phim = np.zeros((3*nb_marker, 1, nb_frame))
        Km = np.zeros((3*nb_marker, nb_segment*12, nb_frame))

        phir = np.zeros((6*nb_segment, 1, nb_frame))
        Kr = np.zeros((6*nb_segment, 12*nb_segment, nb_frame))
        # Definition of the full phir,phim,Kr,Km matrix
        for ind_segment, segment in enumerate(full_segment):
            phir[6*ind_segment:6*(ind_segment+1), :] = segment.get_phir()

            Kr[6*ind_segment:6*(ind_segment+1),
               12*ind_segment:12*(ind_segment+1), :] = segment.get_Kr()

            phim[ind_marker*3:(ind_marker+len(segment.rm))
                 * 3, :] = segment.get_phim()
            # Km[ind_marker*3:(ind_marker+len(segment.rm))*3,
            #    ind_segment*12:(ind_segment+1)*12, :] = \
            #     np.tile(segment.get_Km(), (1, 1, nb_frame))

            temp_DQ = np.zeros((12, 12, nb_frame))
            diag_3 = np.identity(3)[:, :, np.newaxis]
            # uu
            temp_DQ[0:3, 0:3, :] = 2*lambda_r[(ind_segment)*6+0, 0, :]*diag_3
            # uv
            temp_DQ[3:6, 0:3, :] = lambda_r[(ind_segment)*6+1, 0, :]*diag_3
            temp_DQ[0:3, 3:6, :] = lambda_r[(ind_segment)*6+1, 0, :]*diag_3

            temp_DQ[6:9, 0:3, :] = -lambda_r[(ind_segment)*6+1, 0, :]*diag_3
            temp_DQ[0:3, 6:9, :] = -lambda_r[(ind_segment)*6+1, 0, :]*diag_3
            # uw
            temp_DQ[9:12, 0:3, :] = lambda_r[(ind_segment)*6+2, 0, :]*diag_3
            temp_DQ[0:3, 9:12, :] = lambda_r[(ind_segment)*6+2, 0, :]*diag_3
            # vv
            temp_DQ[3:6, 3:6, :] = 2*lambda_r[(ind_segment)*6+3, 0, :]*diag_3
            temp_DQ[6:9, 6:9, :] = 2*lambda_r[(ind_segment)*6+3, 0, :]*diag_3

            temp_DQ[3:6, 6:9, :] = -2*lambda_r[(ind_segment)*6+3, 0, :]*diag_3
            temp_DQ[6:9, 3:6, :] = -2*lambda_r[(ind_segment)*6+3, 0, :]*diag_3
            # vw
            temp_DQ[3:6, 9:12, :] = lambda_r[(ind_segment)*6+4, 0, :]*diag_3
            temp_DQ[9:12, 3:6, :] = lambda_r[(ind_segment)*6+4, 0, :]*diag_3

            temp_DQ[6:9, 9:12, :] = -lambda_r[(ind_segment)*6+4, 0, :]*diag_3
            temp_DQ[9:12, 6:9, :] = -lambda_r[(ind_segment)*6+4, 0, :]*diag_3

            # ww
            temp_DQ[9:12, 9:12, :] = 2*lambda_r[(ind_segment)*6+5, 0, :]*diag_3

            dKlambdardQ[12*ind_segment:12*(ind_segment+1),
                        12*ind_segment:12*(ind_segment+1), :] = np.copy(temp_DQ)

            ind_marker = ind_marker + len(segment.rm)
        # Kinematic constraint
        phik = np.zeros((nb_constraint, 1, nb_frame))
        Kk = np.zeros((nb_constraint, nb_segment*12, nb_frame))
        dKlambdakdQ = np.zeros((12*nb_segment, 12*nb_segment, nb_frame))

        constraint_quantity = 0
        for ind_constraint, constraint in enumerate(full_model):
            phik_temp = constraint.get_phik(
                full_segment[ind_constraint], full_segment[ind_constraint+1])
            Kk_temp = constraint.get_Kk(full_segment[ind_constraint],
                                        full_segment[ind_constraint+1])
            lambda_k_temp = lambda_k[constraint_quantity:constraint_quantity +
                                     constraint.nb_constraint, 0, :]

            phik[constraint_quantity:constraint_quantity+constraint.nb_constraint,
                 :, :] = np.copy(phik_temp[:, np.newaxis, :])

            Kk[constraint_quantity:constraint_quantity+constraint.nb_constraint,
                ind_constraint*12:(ind_constraint+2)*12, :] = Kk_temp

            dKlambdakdQ[ind_constraint*12:(ind_constraint+2)*12, ind_constraint*12:(
                ind_constraint+2)*12, :] = constraint.get_dKlambdakdQ(nb_frame, lambda_k_temp)

            constraint_quantity = constraint_quantity + constraint.nb_constraint

        DKlambdaridQ_Total = dKlambdardQ + dKlambdakdQ

        # Error
        error_phik = np.mean(np.einsum('mnr,ndr->mdr',
                                       np.transpose(phik, (1, 0, 2)), phik))
        error_phim = np.mean(np.einsum('mnr,ndr->mdr',
                                       np.transpose(phim, (1, 0, 2)), phim))
        error_phir = np.mean(np.einsum('mnr,ndr->mdr',
                                       np.transpose(phir, (1, 0, 2)), phir))
        print('Error phim')
        print(error_phim)
        print('Error phir')
        print(error_phir)
        print('Error phik')
        print(error_phik)

        F = np.zeros((nb_segment*12+nb_constraint+6*nb_segment, 1, nb_frame))

        dFdX = np.zeros((nb_segment*12+nb_constraint+6*nb_segment,
                         12*nb_segment+nb_constraint+6*nb_segment,
                         nb_frame))

        F[0:nb_segment*12, :, :] = np.einsum('mnr,ndr->mdr',
                                             np.transpose(Km, (1, 0, 2)),
                                             phim) + np.einsum('mnr,ndr->mdr',
                                                               np.transpose(np.concatenate(
                                                                   (Kk, Kr), axis=0), (1, 0, 2)),
                                                               np.concatenate((lambda_k, lambda_r), axis=0))

        F[nb_segment*12:nb_segment*12+nb_constraint, :, :] = phik
        F[nb_segment*12+nb_constraint:, :, :] = phir

        # dFdX[0:nb_segment*12,
        #      0:nb_segment*12, :] = np.einsum('mnr,ndr->mdr',
        #                                      np.transpose(Km, (1, 0, 2)), Km) + DKlambdaridQ_Total
        dFdX[0:nb_segment*12, 0:nb_segment*12,
             :] = Full_KmT + DKlambdaridQ_Total

        dFdX[0:nb_segment*12,
             nb_segment*12:nb_segment*12+nb_constraint, :] = np.transpose(Kk, (1, 0, 2))
        dFdX[0:nb_segment*12,
             nb_segment*12+nb_constraint:, :] = np.transpose(Kr, (1, 0, 2))

        dFdX[nb_segment*12:nb_segment*12+nb_constraint:,
             0:nb_segment*12, :] = Kk
        dFdX[nb_segment*12+nb_constraint:,
             0:nb_segment*12, :] = Kr

        # inversion matrice OK
        dX = np.einsum('mnr,ndr->mdr', np.linalg.inv(-dFdX.T).T, F)

        # Modification of lambda and segment
        for ind_segment, segment in enumerate(full_segment):
            lvl_seg = ind_segment*12
            segment.u += np.squeeze(dX[lvl_seg+0:lvl_seg+3, 0, :])
            segment.rp += np.squeeze(dX[lvl_seg+3:lvl_seg+6, 0, :])
            segment.rd += np.squeeze(dX[lvl_seg+6:lvl_seg+9, 0, :])
            segment.w += np.squeeze(dX[lvl_seg+9:lvl_seg+12, 0, :])
            segment.Q += np.squeeze(dX[lvl_seg+0:lvl_seg+12, 0, :])

        lambda_k += dX[12*nb_segment:12*nb_segment+nb_constraint, :, :]

        lambda_r += dX[12*nb_segment+nb_constraint:, :, :]

        error = np.sum(np.sqrt(np.einsum('mnr,ndr->mdr',
                                         np.transpose(F, (1, 0, 2)), F)), axis=2)
        print('Error optim')
        print(error)

    # update of all object
    for segment in full_segment:
        segment.update()


def calculate_error(full_segment, full_model):

    nb_frame = full_segment[0].u.shape[1]
    nb_segment = len(full_segment)
    nb_marker = 0
    for segment in full_segment:
        nb_marker = nb_marker + len(segment.rm)

    nb_constraint = 0
    for model in full_model:
        nb_constraint += model.nb_constraint

    # Initialisation
    dKlambdardQ = np.zeros((12*nb_segment, 12*nb_segment, nb_frame))
    lambda_r = np.zeros((nb_segment*6, 1, nb_frame))
    lambda_k = np.zeros((nb_constraint, 1, nb_frame))

    error = 1
    step = 0

    step = step + 1
    print(step)
    # initialisation
    ind_marker = 0
    phim = np.zeros((3*nb_marker, 1, nb_frame))
    Km = np.zeros((3*nb_marker, nb_segment*12, nb_frame))

    phir = np.zeros((6*nb_segment, 1, nb_frame))
    Kr = np.zeros((6*nb_segment, 12*nb_segment, nb_frame))
    # Definition of the full phir,phim,Kr,Km matrix
    for ind_segment, segment in enumerate(full_segment):
        phir[6*ind_segment:6*(ind_segment+1), :] = segment.get_phir()

        Kr[6*ind_segment:6*(ind_segment+1),
            12*ind_segment:12*(ind_segment+1), :] = segment.get_Kr()

        phim[ind_marker*3:(ind_marker+len(segment.rm))
             * 3, :] = segment.get_phim()
        Km[ind_marker*3:(ind_marker+len(segment.rm))*3,
            ind_segment*12:(ind_segment+1)*12, :] = \
            np.tile(segment.get_Km(), (1, 1, nb_frame))

        temp_DQ = np.zeros((12, 12, nb_frame))
        diag_3 = np.identity(3)[:, :, np.newaxis]
        # uu
        temp_DQ[0:3, 0:3, :] = 2*lambda_r[(ind_segment)*6+0, 0, :]*diag_3
        # uv
        temp_DQ[3:6, 0:3, :] = lambda_r[(ind_segment)*6+1, 0, :]*diag_3
        temp_DQ[0:3, 3:6, :] = lambda_r[(ind_segment)*6+1, 0, :]*diag_3

        temp_DQ[6:9, 0:3, :] = -lambda_r[(ind_segment)*6+1, 0, :]*diag_3
        temp_DQ[0:3, 6:9, :] = -lambda_r[(ind_segment)*6+1, 0, :]*diag_3
        # uw
        temp_DQ[9:12, 0:3, :] = lambda_r[(ind_segment)*6+2, 0, :]*diag_3
        temp_DQ[0:3, 9:12, :] = lambda_r[(ind_segment)*6+2, 0, :]*diag_3
        # vv
        temp_DQ[3:6, 3:6, :] = 2*lambda_r[(ind_segment)*6+3, 0, :]*diag_3
        temp_DQ[6:9, 6:9, :] = 2*lambda_r[(ind_segment)*6+3, 0, :]*diag_3

        temp_DQ[3:6, 6:9, :] = -2*lambda_r[(ind_segment)*6+3, 0, :]*diag_3
        temp_DQ[6:9, 3:6, :] = -2*lambda_r[(ind_segment)*6+3, 0, :]*diag_3
        # vw
        temp_DQ[3:6, 9:12, :] = lambda_r[(ind_segment)*6+4, 0, :]*diag_3
        temp_DQ[9:12, 3:6, :] = lambda_r[(ind_segment)*6+4, 0, :]*diag_3

        temp_DQ[6:9, 9:12, :] = -lambda_r[(ind_segment)*6+4, 0, :]*diag_3
        temp_DQ[9:12, 6:9, :] = -lambda_r[(ind_segment)*6+4, 0, :]*diag_3

        # ww
        temp_DQ[9:12, 9:12, :] = 2*lambda_r[(ind_segment)*6+5, 0, :]*diag_3

        dKlambdardQ[12*ind_segment:12*(ind_segment+1),
                    12*ind_segment:12*(ind_segment+1), :] = np.copy(temp_DQ)

        ind_marker = ind_marker + len(segment.rm)
    # Kinematic constraint
    phik = np.zeros((nb_constraint, 1, nb_frame))
    Kk = np.zeros((nb_constraint, nb_segment*12, nb_frame))
    dKlambdakdQ = np.zeros((12*nb_segment, 12*nb_segment, nb_frame))

    constraint_quantity = 0
    for ind_constraint, constraint in enumerate(full_model):
        phik_temp = constraint.get_phik(
            full_segment[ind_constraint], full_segment[ind_constraint+1])
        Kk_temp = constraint.get_Kk(full_segment[ind_constraint],
                                    full_segment[ind_constraint+1])
        lambda_k_temp = lambda_k[constraint_quantity:constraint_quantity +
                                 constraint.nb_constraint, 0, :]

        phik[constraint_quantity:constraint_quantity+constraint.nb_constraint,
             :, :] = np.copy(phik_temp[:, np.newaxis, :])

        Kk[constraint_quantity:constraint_quantity+constraint.nb_constraint,
           ind_constraint*12:(ind_constraint+2)*12, :] = Kk_temp

        dKlambdakdQ[ind_constraint*12:(ind_constraint+2)*12, ind_constraint*12:(
            ind_constraint+2)*12, :] = constraint.get_dKlambdakdQ(nb_frame, lambda_k_temp)

        constraint_quantity = constraint_quantity + constraint.nb_constraint

    DKlambdaridQ_Total = dKlambdardQ + dKlambdakdQ

    # Error
    error_phik = np.mean(np.einsum('mnr,ndr->mdr',
                                   np.transpose(phik, (1, 0, 2)), phik))
    error_phim = np.mean(np.einsum('mnr,ndr->mdr',
                                   np.transpose(phim, (1, 0, 2)), phim))
    error_phir = np.mean(np.einsum('mnr,ndr->mdr',
                                   np.transpose(phir, (1, 0, 2)), phir))
    print('Error phim')
    print(error_phim)
    print('Error phir')
    print(error_phir)
    print('Error phik')
    print(error_phik)
    return[phim, phik, phir]
