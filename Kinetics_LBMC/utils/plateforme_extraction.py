# -*- coding: utf-8 -*-

""".
Created on Fri Feb 22 09:02:39 2019

@author: AdminXPS
"""
import numpy as np
from scipy import signal
#import ezc3d
import os


def plateforme_extraction(acq):
    # Extraction des informations de base
    # Analog frequency

    # Analog extraction
    analog_data = acq['data']['analogs']
    analogs_names = acq['parameters']['ANALOG']['LABELS']['value']

    analog_frq = acq['header']['analogs']['frame_rate']
    point_frq = acq['header']['points']['frame_rate']
    analogs_ind = dict()
    for index_analog, name_analog in enumerate(analogs_names):
        analogs_ind[name_analog] = index_analog

    # Force plateform extraction (COP/force moment at COP)?
    origin = acq['parameters']['FORCE_PLATFORM']['ORIGIN']['value']
    corners = acq['parameters']['FORCE_PLATFORM']['CORNERS']['value']
    list_param = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    nb_plateform = len(origin)//3

    force_plateforms = dict()

    for ind_plateform in range(nb_plateform):
        # found the name of the analog parameters
        list_key_analog = analogs_ind.keys()
        list_name_analog = []
        string_plateforme = str(ind_plateform+1)
        for name_param in list_param:
            for name_analog in list_key_analog:
                if name_param.lower() in name_analog.lower() and\
                        string_plateforme in name_analog.lower():
                    list_name_analog.append(name_analog)

        if len(list_name_analog) > len(list_param):
            temp = [name for name in list_name_analog if (
                'fp' in name.lower() or 'pf' in name.lower())]
            list_name_analog = temp

        force_plateform_temp = dict()

        name_in_data = list_name_analog
        #corner_1 = np.array(corners[ind_plateform*12:ind_plateform*12+3])
        #corner_2 = np.array(corners[ind_plateform*12+3:ind_plateform*12+6])
        #corner_3 = np.array(corners[ind_plateform*12+6:ind_plateform*12+9])
        #corner_4 = np.array(corners[ind_plateform*12+9:ind_plateform*12+12])

        corner_1 = corners[:, 0, ind_plateform]
        corner_2 = corners[:, 1, ind_plateform]
        corner_3 = corners[:, 2, ind_plateform]
        corner_4 = corners[:, 3, ind_plateform]

        X_platform = (corner_1-corner_2) / np.linalg.norm(corner_1-corner_2)
        Y_platform = (corner_2-corner_3) / np.linalg.norm(corner_2-corner_3)
        Z_platform = np.cross(X_platform, Y_platform)

        # Extraction of the axis direction and orientation
        X_pos = np.argmax(abs(X_platform))
        Y_pos = np.argmax(abs(Y_platform))
        Z_pos = np.argmax(abs(Z_platform))
        position_axis = [X_pos, Y_pos, Z_pos]
        position_name = ['x', 'y', 'z']

        X_sign = np.sign(X_platform[X_pos])
        Y_sign = np.sign(Y_platform[Y_pos])
        Z_sign = np.sign(Z_platform[Z_pos])
        mult_sign = np.array([X_sign, Y_sign, Z_sign])

        # force_plateform_temp['origin'] = np.array(
        #    origin[ind_plateform*3:(ind_plateform+1)*3])*mult_sign
        force_plateform_temp['origin'] = origin[:, ind_plateform]*mult_sign

        mid_corner = (corner_1+corner_2+corner_3+corner_4)/4

        force_plateform_temp['center_global'] = mid_corner
        force_plateform_temp['origin_global'] = force_plateform_temp['origin']+mid_corner

        for name_global, param_in_data in zip(list_param, name_in_data):
            if 'x' in param_in_data.lower():
                sign = X_sign
                new_direction = position_name[position_axis[0]]
                final_test_name = name_global.replace('x', new_direction)
            elif 'y' in param_in_data.lower():
                sign = Y_sign
                new_direction = position_name[position_axis[1]]
                final_test_name = name_global.replace('y', new_direction)
            elif 'z' in param_in_data.lower():
                sign = Z_sign
                new_direction = position_name[position_axis[2]]
                final_test_name = name_global.replace('z', new_direction)
            force_plateform_temp[final_test_name] = sign * \
                analog_data[:, analogs_ind[param_in_data], :]
            # Faire une transmutation

        force_plateform_name = 'Fp'+str(ind_plateform+1)
        nbr_analog = force_plateform_temp['Fz'].shape[1]
        # Initialisation
        # name_FCOP = [param+'_COP' for param in list_param if 'F' in param]
        # name_MCOP = [param+'_COP' for param in list_param if 'M' in param]

        # for ind in name_FCOP+name_MCOP:
        # force_plateform_temp[ind] = np.zeros(nbr_analog)

        Mx = force_plateform_temp['Mx']
        My = force_plateform_temp['My']
        Mz = force_plateform_temp['Mz']
        Fx = force_plateform_temp['Fx']
        Fy = force_plateform_temp['Fy']
        Fz = force_plateform_temp['Fz']
        mask = Fz[0, :] < 5
        Mx[0, mask] = 0
        My[0, mask] = 0
        Mz[0, mask] = 0
        Fx[0, mask] = 0
        Fy[0, mask] = 0
        Fz[0, mask] = 0

        # Filter parameter
        Wn = 12/(analog_frq/2)
        b, a = signal.butter(4, Wn, 'lowpass', analog=False, output='ba')
        # Writing of the different component to ease the reading
        Mx = signal.filtfilt(b, a, Mx[0, :])
        My = signal.filtfilt(b, a, My[0, :])
        Mz = signal.filtfilt(b, a, Mz[0, :])
        Fx = signal.filtfilt(b, a, Fx[0, :])
        Fy = signal.filtfilt(b, a, Fy[0, :])
        Fz = signal.filtfilt(b, a, Fz[0, :])

        Xor = force_plateform_temp['origin_global'][0]
        Yor = force_plateform_temp['origin_global'][1]
        # Her it is - because it seems that the origin is expressed in the global frame
        Zor = -force_plateform_temp['origin_global'][2]
        # Calcul COP
        mask = Fz > 5
        Mx_temp = Mx[mask]
        My_temp = My[mask]
        Mz_temp = Mz[mask]
        Fx_temp = Fx[mask]
        Fy_temp = Fy[mask]
        Fz_temp = Fz[mask]
        # initialisation
        X_CoP_temp = np.zeros(nbr_analog)
        Y_CoP_temp = np.zeros(nbr_analog)
        Z_CoP_temp = np.zeros(nbr_analog)

        Mx_CoP_temp = np.zeros(nbr_analog)
        My_CoP_temp = np.zeros(nbr_analog)
        Mz_CoP_temp = np.zeros(nbr_analog)

        # Calcul
        X_CoP_temp[mask] = (-My_temp+Fz_temp*Xor-Fx_temp*Zor)/Fz_temp
        Y_CoP_temp[mask] = (Mx_temp+Fz_temp*Yor-Fy_temp*Zor)/Fz_temp

        Mz_CoP_temp[mask] = (Mz_temp + Fy_temp*(Xor-X_CoP_temp[mask])
                             - Fx_temp*(Yor - Y_CoP_temp[mask]))  # *mask

        CoP = np.array([X_CoP_temp, Y_CoP_temp, Z_CoP_temp])
        M_CoP = np.array([Mx_CoP_temp, My_CoP_temp, Mz_CoP_temp])
        F_CoP = np.array([Fx*mask, Fy*mask, Fz*mask])
        M = np.array([Mx*mask, My*mask, Mz*mask])
        F = np.array([Fx*mask, Fy*mask, Fz*mask])
        Or = np.array([Xor, Yor, Zor])

        force_plateform_temp['CoP'] = CoP[:,
                                          0::int(round(analog_frq/point_frq))]
        force_plateform_temp['M_CoP'] = M_CoP[:,
                                              0::int(round(analog_frq/point_frq))]
        force_plateform_temp['F_CoP'] = F_CoP[:,
                                              0::int(round(analog_frq/point_frq))]

        force_plateform_temp['origin_plateform'] = np.tile(
            Or[:, np.newaxis], (1, force_plateform_temp['CoP'].shape[1]))

        force_plateform_temp['M'] = M[:, 0::int(round(analog_frq/point_frq))]
        force_plateform_temp['F'] = F[:, 0::int(round(analog_frq/point_frq))]

        force_plateforms[force_plateform_name] = force_plateform_temp

    return force_plateforms


if __name__ == '__main__':

    filename = os.path.join('.', 'data', 'dev_tool_box.c3d')
    # Extraction du c3d
    #acq = ezc3d.c3d(filename)
    #force_plateforms = plateforme_extraction(acq)
