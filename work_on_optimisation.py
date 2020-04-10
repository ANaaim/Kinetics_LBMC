# -*- coding: utf-8 -*-
from multi_body_optimisation import multi_body_optimisation as multi_body_optimisation
from multi_body_optimisation import calculate_error as calculate_error
import pdb
from KinematicChain import KinematicChain as KinematicChain
from HomogeneousMatrix import HomogeneousMatrix as HomogeneousMatrix
import r2mobile
from matplotlib import pyplot as plt
import points_treatment

from Segment import Segment as Segment
import joint_model
from norm_vector import norm_vector as norm_vector
import numpy as np
import os
import ezc3d

from leardini_simplified import leardini_simplified as leardini_simplified
# ---Extraction en utilisant EzC3D---
filename_static = os.path.join('.', 'data', 'static_foot.c3d')
filename_dynamic = os.path.join('.', 'data', 'dynamic_foot.c3d')

# Extraction du c3d
acq_static = ezc3d.c3d(filename_static)
# Point ind name extraction
points_names_static = acq_static['parameters']['POINT']['LABELS']['value']
# Extraction du c3d
acq_dynamic = ezc3d.c3d(filename_dynamic)
# Point ind name extraction
points_names_dynamic = acq_dynamic['parameters']['POINT']['LABELS']['value']

# Point name indice extraction
# points_ind_static = dict()
# for index_point, name_point in enumerate(points_names_static):
#    points_ind_static[name_point] = index_point
# Point name indice extraction
# points_ind_dynamic = dict()
# for index_point, name_point in enumerate(points_names_dynamic):
#    points_ind_dynamic[name_point] = index_point

# ==============================================================================
# TODO faire un exemple avec une base sur une statique à la fois segment et model
# ==============================================================================


#points_static = points_treatment_validation_QUT.points_treatment_validation_foot(acq_static)
#points_dynamic = points_treatment_validation_QUT.points_treatment_validation_foot(acq_dynamic)

points_static = points_treatment.points_treatment(
    acq_static, 10, unit_point='m')
points_dynamic = points_treatment.points_treatment(
    acq_dynamic, 10, unit_point='m')

[segment_foot_static, segment_tibia_static, segment_thigh_static,
    segment_pelvis_static] = leardini_simplified(points_static, points_names_static)

[segment_foot_dynamic, segment_tibia_dynamic, segment_thigh_dynamic,
    segment_pelvis_dynamic] = leardini_simplified(points_dynamic, points_names_dynamic)
[segment_foot_dynamic2, segment_tibia_dynamic2, segment_thigh_dynamic2,
    segment_pelvis_dynamic2] = leardini_simplified(points_dynamic, points_names_dynamic)


segment_foot_final = Segment.fromSegment(
    segment_foot_dynamic, segment_static=segment_foot_static)

segment_tibia_final = Segment.fromSegment(
    segment_tibia_dynamic, segment_static=segment_tibia_static)

segment_thigh_final = Segment.fromSegment(
    segment_thigh_dynamic, segment_static=segment_thigh_static)

segment_pelvis_final = Segment.fromSegment(
    segment_pelvis_dynamic, segment_static=segment_pelvis_static)


full_segment = [segment_foot_final, segment_tibia_final,
                segment_thigh_final, segment_pelvis_final]
copy_full_segment = [segment_foot_dynamic2, segment_tibia_dynamic2,
                     segment_thigh_dynamic2, segment_pelvis_dynamic2]
full_segment_hip = [segment_thigh_dynamic, segment_pelvis_dynamic]
full_segment_knee = [segment_tibia_dynamic, segment_thigh_dynamic]
full_segment_ankle = [segment_foot_dynamic, segment_tibia_dynamic]
full_segment_test = [segment_foot_dynamic, segment_tibia_dynamic,
                     segment_pelvis_dynamic]  # , segment_thigh_dynamic]

model_from_static = True
if model_from_static:
    ankle_model = joint_model.universal_model(
        segment_foot_static, segment_tibia_static, 'u', 'w')
    #ankle_model = joint_model.hinge_model(segment_foot_static, segment_tibia_static)
    knee_model = joint_model.hinge_model(
        segment_tibia_static, segment_thigh_static)
    hip_model = joint_model.spherical_model(segment_thigh_static, segment_pelvis_static,
                                            segment_thigh_static.rp, segment_thigh_static.rp)
else:
    ankle_model = joint_model.spherical_model(segment_foot_dynamic, segment_tibia_dynamic,
                                              segment_foot_dynamic.rp, segment_tibia_dynamic.rd)
    knee_model = joint_model.spherical_model(segment_tibia_dynamic, segment_thigh_dynamic,
                                             segment_tibia_dynamic.rp, segment_thigh_dynamic.rd)
    hip_model = joint_model.spherical_model(segment_thigh_dynamic, segment_pelvis_dynamic,
                                            segment_thigh_dynamic.rp, segment_thigh_dynamic.rp)
pdb.set_trace()

full_model = [ankle_model, knee_model, hip_model]
full_model_test = [joint_model.no_model(), joint_model.no_model(),
                   joint_model.no_model()]
multi_body_optimisation(full_segment, full_model)

full_model_hip = [hip_model]
full_model_knee = [knee_model]
full_model_ankle = [ankle_model]
full_model_test = [joint_model.no_model(), joint_model.no_model()]
#full_model = [joint_model.no_model(), joint_model.no_model(), joint_model.no_model()]
#multi_body_optimisation(full_segment_test, full_model_test)

#multi_body_optimisation([segment_foot_dynamic], [])
#multi_body_optimisation([segment_tibia_dynamic], [])
#multi_body_optimisation([segment_thigh_dynamic], [])
#multi_body_optimisation([segment_pelvis_dynamic], [])
#multi_body_optimisation(full_segment, full_model)
#multi_body_optimisation(copy_full_segment, full_model)

phi_zeros = HomogeneousMatrix.fromHomo(
    np.zeros((4, 4, full_segment[0].u.shape[1])))
# full_segment = [segment_foot, segment_tibia, segment_thigh, segment_pelvis]
# phi_ext express at the origin
phi_ext = [phi_zeros, phi_zeros, phi_zeros, phi_zeros]
# Name of the joint
name_Joint = ['Ankle', 'Knee', 'Hip']
# Euler sequences associated to each name_Joint
rot = [r2mobile.zyx, r2mobile.zxy, r2mobile.zxy]
name_rot = ['zyx', 'zxy', 'zxy']
# Point of calcul of the Moment and Force
point_limb = [segment_tibia_final.get_distal_frame_glob(),
              segment_thigh_final.get_distal_frame_glob(),
              segment_thigh_final.get_proximal_frame_glob()]
# Frame of expression of Moment and Force if not in JCS
frame_limb = [segment_tibia_final.Tdist,
              segment_thigh_final.Tdist, segment_pelvis_final.Tprox]
# Side
side = ['Right']

test_multiseg = KinematicChain(full_segment, phi_ext,
                               name_Joint, name_rot,
                               point_limb, frame_limb,
                               'JCS')
test_multiseg_no_optim = KinematicChain(copy_full_segment, phi_ext,
                                        name_Joint, name_rot,
                                        point_limb, frame_limb,
                                        'JCS')


plt.subplot(131)
plt.plot(test_multiseg.euler_glob['Pelvis'][0, :])
plt.plot(test_multiseg_no_optim.euler_glob['Pelvis'][0, :])
plt.title('Pelvis')
plt.subplot(132)
plt.plot(test_multiseg.euler_glob['Pelvis'][1, :])
plt.plot(test_multiseg_no_optim.euler_glob['Pelvis'][1, :])
plt.title('Pelvis')
plt.subplot(133)
plt.plot(test_multiseg.euler_glob['Pelvis'][2, :])
plt.plot(test_multiseg_no_optim.euler_glob['Pelvis'][2, :])
plt.title('Pelvis')
plt.show()

# Tracer des courbes
plt.subplot(331)
plt.plot(test_multiseg.euler_rel['Hip'][0, :], label='Autom')
plt.plot(test_multiseg_no_optim.euler_rel['Hip'][0, :], label='No_optim')
plt.title('Hip')

plt.subplot(332)
plt.plot(test_multiseg.euler_rel['Hip'][1, :], label='Autom')
plt.plot(test_multiseg_no_optim.euler_rel['Hip'][1, :], label='No_optim')
plt.legend()
plt.title('Hip')

plt.subplot(333)
plt.plot(test_multiseg.euler_rel['Hip'][2, :], label='Autom')
plt.plot(test_multiseg_no_optim.euler_rel['Hip'][2, :], label='No_optim')
plt.title('Hip')


plt.subplot(334)
plt.plot(test_multiseg.euler_rel['Knee'][0, :], label='Autom')
plt.plot(test_multiseg_no_optim.euler_rel['Knee'][0, :], label='Autom')
plt.title('Knee')


plt.subplot(335)
plt.plot(test_multiseg.euler_rel['Knee'][1, :], label='Autom')
plt.plot(test_multiseg_no_optim.euler_rel['Knee'][1, :], label='No_optim')
plt.title('Knee')


plt.subplot(336)
plt.plot(test_multiseg.euler_rel['Knee'][2, :], label='Autom')
plt.plot(test_multiseg_no_optim.euler_rel['Knee'][2, :], label='No_Optim')
plt.title('Knee')


plt.subplot(337)
plt.plot(test_multiseg.euler_rel['Ankle'][0, :], label='Autom')
plt.plot(test_multiseg_no_optim.euler_rel['Ankle'][0, :], label='No_Optim')
plt.title('Ankle')


plt.subplot(338)
plt.plot(test_multiseg.euler_rel['Ankle'][1, :], label='Autom')
plt.plot(test_multiseg_no_optim.euler_rel['Ankle'][1, :], label='No_Optim')
plt.title('Ankle')


plt.subplot(339)
plt.plot(test_multiseg.euler_rel['Ankle'][2, :], label='Autom')
plt.plot(test_multiseg_no_optim.euler_rel['Ankle'][2, :], label='No_Optim')
plt.title('Ankle')
plt.show()


# copy des infos pour les points
new_list = points_names_dynamic.copy()
new_array = acq_dynamic['data']['points'].copy()*1000
nb_frame = acq_dynamic['data']['points'].shape[2]

for segment in full_segment:
    name_segment = segment.segment_name
    for ind_rm in range(len(segment.nm_list)):
        name_marker = name_segment + str(ind_rm)
        print(name_marker)

        new_list.append(name_marker)
        new_point = np.zeros((4, 1, nb_frame))

        temp = np.dot(
            segment.nm_list[ind_rm].T, segment.Q)*1000
        new_point[0, 0, :] = temp[0, :]
        new_point[1, 0, :] = -temp[2, :]
        new_point[2, 0, :] = temp[1, :]
        new_point[3, 0, :] = 1

        print('tata')
        new_array = np.append(new_array, new_point, axis=1)
    list_point_to_add = [segment.rp+0.1*segment.u,
                         segment.rp, segment.rd, segment.rd+0.1*segment.w]
    list_name = ['u', 'rp', 'rd', 'w']
    for ind_point, point in enumerate(list_point_to_add):
        name_point = list_name[ind_point] + '_'+name_segment
        print(name_point)
        new_list.append(name_point)
        new_point = np.zeros((4, 1, nb_frame))
        temp = point * 1000
        new_point[0, 0, :] = temp[0, :]
        new_point[1, 0, :] = -temp[2, :]
        new_point[2, 0, :] = temp[1, :]
        new_point[3, 0, :] = 1

        new_array = np.append(new_array, new_point, axis=1)

# acq_dynamic['data']['points'] = new_array
# acq_dynamic['parameters']['POINT']['LABELS']['value'] = new_list


c3d = ezc3d.c3d()

# Fill it with random data
c3d['parameters']['POINT']['RATE']['value'] = [nb_frame]
c3d['parameters']['POINT']['LABELS']['value'] = new_list
c3d['data']['points'] = new_array

c3d.write('test_ezc3cd.c3d')
