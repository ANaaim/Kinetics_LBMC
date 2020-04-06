# -*- coding: utf-8 -*-
from multi_body_optimisation import multi_body_optimisation as multi_body_optimisation
import pdb
from KinematicChain import KinematicChain as KinematicChain
from HomogeneousMatrix import HomogeneousMatrix as HomogeneousMatrix
import r2mobile
from matplotlib import pyplot as plt
import points_treatment
import points_treatment_validation_QUT
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

points_static = points_treatment.points_treatment(acq_static,10,unit_point='m')
points_dynamic = points_treatment.points_treatment(acq_dynamic,10,unit_point='m')

[segment_foot_static, segment_tibia_static, segment_thigh_static,
    segment_pelvis_static] = leardini_simplified(points_static, points_names_static)

[segment_foot_dynamic, segment_tibia_dynamic, segment_thigh_dynamic,
    segment_pelvis_dynamic] = leardini_simplified(points_dynamic, points_names_dynamic)

full_segment = [segment_foot_dynamic, segment_tibia_dynamic,
                segment_thigh_dynamic, segment_pelvis_dynamic]


# Creation of a the external force (here no forces)
phi_zeros = HomogeneousMatrix.fromHomo(np.zeros((4, 4, full_segment[0].u.shape[1])))
phi_ext = [phi_zeros, phi_zeros, phi_zeros, phi_zeros]
# Name of the joint
name_Joint = ['Ankle', 'Knee', 'Hip']
# Euler sequences associated to each name_Joint
name_rot = ['zyx', 'zxy', 'zxy']
# Point of calcul of the Moment and Force
point_limb = [segment_tibia_dynamic.get_distal_frame_glob(),
              segment_thigh_dynamic.get_distal_frame_glob(),
              segment_thigh_dynamic.get_proximal_frame_glob()]
# Frame of expression of Moment and Force if not in JCS
frame_limb = [segment_tibia_dynamic.Tdist,
              segment_thigh_dynamic.Tdist, segment_pelvis_dynamic.Tprox]

test_multiseg = KinematicChain(full_segment, phi_ext,
                               name_Joint, name_rot,
                               point_limb, frame_limb,
                               'JCS')


plt.subplot(131)
plt.plot(test_multiseg.euler_glob['Pelvis'][0, :])
plt.title('Pelvis')
plt.subplot(132)
plt.plot(test_multiseg.euler_glob['Pelvis'][1, :])
plt.title('Pelvis')
plt.subplot(133)
plt.plot(test_multiseg.euler_glob['Pelvis'][2, :])
plt.title('Pelvis')
plt.show()

# Tracer des courbes
plt.subplot(331)
plt.plot(test_multiseg.euler_rel['Hip'][0, :])
plt.title('Hip')

plt.subplot(332)
plt.plot(test_multiseg.euler_rel['Hip'][1, :])
plt.legend()
plt.title('Hip')

plt.subplot(333)
plt.plot(test_multiseg.euler_rel['Hip'][2, :])
plt.title('Hip')


plt.subplot(334)
plt.plot(test_multiseg.euler_rel['Knee'][0, :])
plt.title('Knee')


plt.subplot(335)
plt.plot(test_multiseg.euler_rel['Knee'][1, :])
plt.title('Knee')


plt.subplot(336)
plt.plot(test_multiseg.euler_rel['Knee'][2, :])
plt.title('Knee')


plt.subplot(337)
plt.plot(test_multiseg.euler_rel['Ankle'][0, :])
plt.title('Ankle')


plt.subplot(338)
plt.plot(test_multiseg.euler_rel['Ankle'][1, :])
plt.title('Ankle')


plt.subplot(339)
plt.plot(test_multiseg.euler_rel['Ankle'][2, :])
plt.title('Ankle')
plt.show()