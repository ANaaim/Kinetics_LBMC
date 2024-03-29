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
    # Weight Matrix
    W = np.zeros((3*nb_marker, 3*nb_marker, nb_frame))
    Km = np.zeros((3*nb_marker, nb_segment*12, nb_frame))
    ind_marker = 0
    for ind_segment, segment in enumerate(full_segment):
        Km[ind_marker*3:(ind_marker+len(segment.rm))*3,
           ind_segment*12:(ind_segment+1)*12, :] = \
            np.tile(segment.get_Km(), (1, 1, nb_frame))
        W[ind_marker*3:(ind_marker+len(segment.rm))*3,
          ind_marker*3:(ind_marker+len(segment.rm))*3, :] = \
            segment.get_Weight_Matrix()
        ind_marker = ind_marker + len(segment.rm)

    temp_KmT = np.einsum('mnr,ndr->mdr', np.transpose(Km, (1, 0, 2)), W)
    Full_KmT = np.einsum('mnr,ndr->mdr', temp_KmT, Km)
    error = 1
    step = 0
    # Value to be sure to enter the loop and large former error to be sure to be above any level that could be
    # found.
    error_evolution = 1
    former_error = 1000000000000000
    while error_evolution > 10**-3 and step < max_step:
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
        #import pdb
        # pdb.set_trace()
        # We know that all phim that are NaN are supposed to be part where no point have been recorded.
        # As a result
        phim = np.nan_to_num(phim)
        constraint_quantity = 0
        for ind_constraint, constraint in enumerate(full_model):
            ind_prox = constraint.proximal_indice
            ind_dist = constraint.distal_indice

            phik_temp = constraint.get_phik(
                full_segment[ind_dist], full_segment[ind_prox])

            Kk_temp = constraint.get_Kk(full_segment[ind_dist],
                                        full_segment[ind_prox])

            lambda_k_temp = lambda_k[constraint_quantity:constraint_quantity +
                                     constraint.nb_constraint, 0, :]

            phik[constraint_quantity:constraint_quantity+constraint.nb_constraint,
                 :, :] = np.copy(phik_temp[:, np.newaxis, :])
            # Division of Kk and DklambakdQ
            Kk[constraint_quantity:constraint_quantity+constraint.nb_constraint,
                ind_prox*12:(ind_prox+1)*12, :] = Kk_temp[:, 12:24, :]
            Kk[constraint_quantity:constraint_quantity+constraint.nb_constraint,
                ind_dist*12:(ind_dist+1)*12, :] = Kk_temp[:, 0:12, :]

            dKlambdakdQ_temp = constraint.get_dKlambdakdQ(
                nb_frame, lambda_k_temp)

            dKlambdakdQ[ind_constraint*12:(ind_constraint+2)*12, ind_prox*12:(
                ind_prox+1)*12, :] = dKlambdakdQ_temp[:, 12:24, :]
            dKlambdakdQ[ind_constraint*12:(ind_constraint+2)*12, ind_dist*12:(
                ind_dist+1)*12, :] = dKlambdakdQ_temp[:, 0:12, :]

            constraint_quantity = constraint_quantity + constraint.nb_constraint

        DKlambdaridQ_Total = dKlambdardQ + dKlambdakdQ

        # Error
        error_phik = np.mean(np.einsum('mnr,ndr->mdr',
                                       np.transpose(phik, (1, 0, 2)), phik))
        temp_error_phim = np.einsum(
            'mnr,ndr->mdr', np.transpose(phim, (1, 0, 2)), W)
        error_phim = np.mean(np.einsum('mnr,ndr->mdr', temp_error_phim, phim))
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
        # C'est pas déjà calculer çà ??? ca à l'air....
        # temp_Km_W_phim = np.einsum(
        #    'mnr,ndr->mdr', np.transpose(Km, (1, 0, 2)), W)
        temp_Km_W_phim = temp_KmT
        F[0:nb_segment*12, :, :] = np.einsum('mnr,ndr->mdr',
                                             temp_Km_W_phim,
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
        # New error calculation
        # TODO : Est ce que Fm permet bien uniquement le calcul de l'erreur phim.
        F_m = F[0:nb_segment*12, :, :]
        F_k = F[nb_segment*12:nb_segment*12+nb_constraint, :, :]
        F_r = F[nb_segment*12+nb_constraint:, :, :]
        error_phim = np.sum(np.sqrt(np.einsum('mnr,ndr->mdr',
                                              np.transpose(F_m, (1, 0, 2)), F_m)), axis=2)
        error_phim = np.sqrt(error_phim)/(phim.shape[0]/3*phim.shape[2])

        error_phik = np.sum(np.sqrt(np.einsum('mnr,ndr->mdr',
                                              np.transpose(F_k, (1, 0, 2)), F_k)), axis=2)
        error_phik = np.sqrt(error_phik)/(phik.shape[0]/3*phik.shape[2])

        error_phir = np.sum(np.sqrt(np.einsum('mnr,ndr->mdr',
                                              np.transpose(F_r, (1, 0, 2)), F_r)), axis=2)
        error_phir = np.sqrt(error_phir)/(phir.shape[0]/3*phir.shape[2])

        error = error_phim/1000 + error_phik + error_phir
        print('Error optim')
        print(error)
        error_evolution = abs(former_error-error)/former_error
        print('percentage relative to former error')
        print(error_evolution)
        former_error = error
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


# TODO : function that from a list of segment and
def multi_body_optimisation_scipy(full_Q, full_segment, full_model):

    # Full_segment to vector Q
    nb_segment = len(full_segment)
    nb_frame = full_segment[0].rp.shape[1]
    #full_Q = np.zeros(12*nb_segment*nb_frame)
    # for ind_segment, segment in enumerate(full_segment):
    #    base_point = ind_segment*12*nb_frame
    #    full_Q[base_point:base_point+3 * nb_frame] = np.ravel(segment.u)
    #    full_Q[base_point+3 * nb_frame:base_point+6 *
    #           nb_frame] = np.reshape(segment.rp, (3*nb_frame,))
    #    full_Q[base_point+6 * nb_frame:base_point+9 *
    #           nb_frame] = np.reshape(segment.rd, (3*nb_frame,))
    #    full_Q[base_point+9 * nb_frame:base_point+12 *
    #           nb_frame] = np.reshape(segment.w, (3*nb_frame,))
    new_full_segment = list()
    for ind_segment, segment in enumerate(full_segment):
        base_point = ind_segment*12*nb_frame
        #u_temp = np.zeros((3, nb_frame))
        u_temp = np.reshape(
            full_Q[base_point:base_point+3 * nb_frame], (3, nb_frame))
        #rp_temp = np.zeros((3, nb_frame))
        rp_temp = np.reshape(
            full_Q[base_point+3 * nb_frame:base_point+6 * nb_frame], (3, nb_frame))
        #rd_temp = np.zeros((3, nb_frame))
        rd_temp = np.reshape(
            full_Q[base_point+6 * nb_frame:base_point+9 * nb_frame], (3, nb_frame))
        #w_temp = np.zeros((3, nb_frame))
        w_temp = np.reshape(
            full_Q[base_point+9 * nb_frame:base_point+12 * nb_frame], (3, nb_frame))

        segment.u = u_temp
        segment.rp = rp_temp
        segment.rd = rd_temp
        segment.w = w_temp
        segment.update()
        new_full_segment.append(segment)
    full_segment = new_full_segment
    print(full_Q)
    nb_frame = full_segment[0].u.shape[1]
    nb_segment = len(full_segment)
    nb_marker = 0
    for segment in full_segment:
        nb_marker = nb_marker + len(segment.rm)

    nb_constraint = 0
    for model in full_model:
        nb_constraint += model.nb_constraint

    # Construction des éléments constants
    # Weight Matrix
    W = np.zeros((3*nb_marker, 3*nb_marker, nb_frame))
    ind_marker = 0
    for ind_segment, segment in enumerate(full_segment):
        W[ind_marker*3:(ind_marker+len(segment.rm))*3,
          ind_marker*3:(ind_marker+len(segment.rm))*3, :] = \
            segment.get_Weight_Matrix()
        ind_marker = ind_marker + len(segment.rm)

    # Value to be sure to enter the loop and large former error to be sure to be above any level that could be
    # found.
    # initialisation
    ind_marker = 0
    phim = np.zeros((3*nb_marker, 1, nb_frame))

    phir = np.zeros((6*nb_segment, 1, nb_frame))
    # Definition of the full phir,phim,Kr,Km matrix

    for ind_segment, segment in enumerate(full_segment):
        phir[6*ind_segment:6*(ind_segment+1), :] = segment.get_phir()

        phim[ind_marker*3:(ind_marker+len(segment.rm))
             * 3, :] = segment.get_phim()

        ind_marker = ind_marker + len(segment.rm)
    # Kinematic constraint
    phik = np.zeros((nb_constraint, 1, nb_frame))

    #import pdb
    # pdb.set_trace()
    # We know that all phim that are NaN are supposed to be part where no point have been recorded.
    # As a result
    phim = np.nan_to_num(phim)
    constraint_quantity = 0
    for ind_constraint, constraint in enumerate(full_model):
        ind_prox = constraint.proximal_indice
        ind_dist = constraint.distal_indice

        phik_temp = constraint.get_phik(
            full_segment[ind_dist], full_segment[ind_prox])

        phik[constraint_quantity:constraint_quantity+constraint.nb_constraint,
             :, :] = np.copy(phik_temp[:, np.newaxis, :])
    # Error
    error_phik = np.mean(np.einsum('mnr,ndr->mdr',
                                   np.transpose(phik, (1, 0, 2)), phik))
    temp_error_phim = np.einsum(
        'mnr,ndr->mdr', np.transpose(phim, (1, 0, 2)), W)
    error_phim = np.mean(np.einsum('mnr,ndr->mdr', temp_error_phim, phim))
    error_phir = np.mean(np.einsum('mnr,ndr->mdr',
                                   np.transpose(phir, (1, 0, 2)), phir))
    print(error_phim+error_phik+error_phir)
    return error_phim + error_phik + error_phir


def sum_Q(Q, full, toto):
    toto = Q*Q
    print(toto.sum())
    return toto.sum()
