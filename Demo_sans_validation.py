# -*- coding: utf-8 -*-
import numpy as np
import os
import ezc3d
from matplotlib import pyplot as plt

import r2mobile
from KinematicChain import KinematicChain as KinematicChain
from HomogeneousMatrix import HomogeneousMatrix as HomogeneousMatrix
from plateforme_extraction import plateforme_extraction as plateforme_extraction
import points_treatment_validation_QUT

from Segment import Segment as Segment
from norm_vector import norm_vector as norm_vector

#from multi_body_optimisation import multi_body_optimisation as multi_body_optimisation
#import joint_model
# ---Extraction en utilisant EzC3D---
filename = os.path.join('.', 'data', 'data_QUT.c3d')

# Extraction du c3d
acq = ezc3d.c3d(filename)
# Point ind name extraction
points_names = acq['parameters']['POINT']['LABELS']['value']
sexe = 'M'
weight = 728/9.81
# Point name indice extraction
points_ind = dict()
for index_point, name_point in enumerate(points_names):
    points_ind[name_point] = index_point

# Donner les axes
# Faire une fonction automatique de traitements de points ==> l'utilisateur donne l'unité du fichier
# et l'ordre des axe antero-post, medio-lat, vertical avec + ou - en fonction du sens
# et ca met en forme pour tous le code.
points = points_treatment_validation_QUT.points_treatment_validation_QUT(acq)

# Value for ploting
points_plotting = acq['data']['points'][0:3, :, :]

# ==============================================================================
# Creation of the different segment static
# ==============================================================================


# ==============================================================================
# Creation of the different segment dynamic
# ==============================================================================
u5 = norm_vector(points[:, points_ind['rightU_5'], :] -
                 points[:, points_ind['rightrp_5'], :])
rp5 = points[:, points_ind['rightrp_5'], :]
rd5 = points[:, points_ind['rightrd_5'], :]
w5 = norm_vector(points[:, points_ind['rightW_5'], :] -
                 points[:, points_ind['rightrd_5'], :])
RASI = points[:, points_ind['R_ASIS'], :]
LASI = points[:, points_ind['L_ASIS'], :]
SACR = points[:, points_ind['M_SACR'], :]
rm5 = [RASI, LASI, SACR]

u4 = norm_vector(points[:, points_ind['rightU_4'], :] -
                 points[:, points_ind['rightrp_4'], :])
rp4 = points[:, points_ind['rightrp_4'], :]
rd4 = points[:, points_ind['rightrd_4'], :]
w4 = norm_vector(points[:, points_ind['rightW_4'], :] -
                 points[:, points_ind['rightrd_4'], :])
GTRO = points[:, points_ind['R_Fake_GTRO'], :]
KNEE = points[:, points_ind['R_KNEE'], :]
KJC = points[:, points_ind['rightrd_4'], :]
rm4 = [GTRO, KNEE, KJC]


u3 = norm_vector(points[:, points_ind['rightU_3'], :] -
                 points[:, points_ind['rightrp_3'], :])
rp3 = points[:, points_ind['rightrp_3'], :]
rd3 = points[:, points_ind['rightrd_3'], :]
w3 = norm_vector(points[:, points_ind['rightW_3'], :] -
                 points[:, points_ind['rightrd_3'], :])
TIBA = points[:, points_ind['R_TIBA'], :]
MALE = points[:, points_ind['R_MALE'], :]
rm3 = [KNEE, TIBA, MALE]


u2 = norm_vector(points[:, points_ind['rightU_2'], :] -
                 points[:, points_ind['rightrp_2'], :])
rp2 = points[:, points_ind['rightrp_2'], :]
rd2 = points[:, points_ind['rightrd_2'], :]
w2 = norm_vector(points[:, points_ind['rightW_2'], :] -
                 points[:, points_ind['rightrd_2'], :])
HEEL = points[:, points_ind['R_HEEL'], :]
META = points[:, points_ind['R_META'], :]
rm2 = [MALE, HEEL, META]

# Pour info : Comment faire un produit en croix
produit_en_croix = np.cross(u5, w5, axis=0)

# ==============================================================================
# Creation Segment
# ==============================================================================
# TODO faire la version defaut de la creation ou Tprox et Tdist sont Buv et Buw
segment_pelvis = Segment(u5, rp5, rd5, w5, rm5,
                         'Buv', 'Bwu',
                         'Pelvis', sexe=sexe, weight=weight, rigid_parameter=True)
# mettre possibilité de rien mettre pour weight sex et type
segment_thigh = Segment(u4, rp4, rd4, w4, rm4,
                        'Buv', 'Bwu',
                        'Thigh', sexe=sexe, weight=weight, rigid_parameter=True)

segment_tibia = Segment(u3, rp3, rd3, w3, rm3,
                        'Buv', 'Bwu',
                        'Tibia', sexe=sexe, weight=weight, rigid_parameter=True)

segment_foot = Segment(u2, rp2, rd2, w2, rm2,
                       'Buw', 'Bwu',
                       'Foot', sexe=sexe, weight=weight, rigid_parameter=True)
# Creation d'un segment sans masse sans inertie on ne lui met pas de poids
segment_foot_sans_inertie = Segment(u2, rp2, rd2, w2, rm2,
                                    'Buw', 'Bwu',
                                    'Foot', rigid_parameter=True)

nb_frame = segment_foot.Tprox.T_homo.shape[2]
# ==============================================================================
# Extraction F, M, Cop plateforme de force
# ==============================================================================
test_plat = plateforme_extraction(acq)

FP1_origin = np.tile(test_plat['Fp1']['origin_global'][:, np.newaxis], (1, nb_frame))/1000
FP1_corner = np.tile(test_plat['Fp1']['origin_global'][:, np.newaxis], (1, nb_frame))/1000

# From mm to m (for the CoP and the moment)
FP1_COP = test_plat['Fp1']['CoP']/1000
FP1_F = test_plat['Fp1']['F_CoP']
FP1_M = test_plat['Fp1']['M_CoP']/1000

# ISB convention
FP1_COP_ISB = np.zeros(FP1_COP.shape)
FP1_COP_ISB[0, :] = FP1_COP[0, :]
FP1_COP_ISB[1, :] = FP1_COP[2, :]
FP1_COP_ISB[2, :] = -FP1_COP[1, :]

FP1_F_ISB = np.zeros(FP1_F.shape)
FP1_F_ISB[0, :] = FP1_F[0, :]
FP1_F_ISB[1, :] = FP1_F[2, :]
FP1_F_ISB[2, :] = -FP1_F[1, :]

FP1_M_ISB = np.zeros(FP1_M.shape)
FP1_M_ISB[0, :] = FP1_M[0, :]
FP1_M_ISB[1, :] = FP1_M[2, :]
FP1_M_ISB[2, :] = -FP1_M[1, :]


# definition of
X_glob = np.tile(np.array([1, 0, 0])[:, np.newaxis], (1, nb_frame))
Y_glob = np.tile(np.array([0, 1, 0])[:, np.newaxis], (1, nb_frame))
Z_glob = np.tile(np.array([0, 0, 1])[:, np.newaxis], (1, nb_frame))

seg_fp1 = Segment(X_glob, FP1_COP_ISB, -Y_glob, Z_glob, rm5,
                  'Buv', 'Bwu', 'plateform')


# Transport of the phi plateform to the origin ==> a mettre dans une fonction ?
transport_plat = np.identity(4)
transport_plat = transport_plat[:, :, np.newaxis]
transport_plat = np.tile(transport_plat, (1, 1, nb_frame))
transport_plat[0:3, 3, :] = seg_fp1.Tprox.T_homo[0:3, 3, :]
transport_plat = HomogeneousMatrix.fromHomo(transport_plat)

FP1_F_ISB = -FP1_F_ISB
FP1_M_ISB = -FP1_M_ISB

phi_plat = np.zeros((4, 4, nb_frame))
phi_plat[0:3, 3, :] = FP1_F_ISB
phi_plat[3, 0:3, :] = -FP1_F_ISB

phi_plat[0, 1, :] = -FP1_M_ISB[2, :]
phi_plat[1, 0, :] = FP1_M_ISB[2, :]

phi_plat[0, 2, :] = FP1_M_ISB[1, :]
phi_plat[2, 0, :] = -FP1_M_ISB[1, :]

phi_plat[1, 2, :] = -FP1_M_ISB[0, :]
phi_plat[2, 1, :] = FP1_M_ISB[0, :]

phi_plat = transport_plat * (HomogeneousMatrix.fromHomo(phi_plat)*transport_plat.transpose())
# ==============================================================================
# Optimisation multi segmentaire
# ==============================================================================
full_segment = [segment_foot, segment_tibia, segment_thigh, segment_pelvis]

#ankle_model = joint_model.universal_model(segment_foot, segment_tibia, 'u', 'w')
#knee_model = joint_model.hinge_model(segment_tibia, segment_thigh)
#hip_model = joint_model.spherical_model(segment_thigh, segment_pelvis,
                                        #segment_thigh.rp, segment_thigh.rp)

#full_model = [ankle_model, knee_model, hip_model]

#multi_body_optimisation(full_segment, full_model)
# poids deja appliqué a mettre en commentaire quelque part
# ==============================================================================
# Inverse Dynamics
# ==============================================================================
# Kinematic
# Developpement
phi_zeros = HomogeneousMatrix.fromHomo(np.zeros((4, 4, nb_frame)))
# full_segment = [segment_foot, segment_tibia, segment_thigh, segment_pelvis]
# phi_ext express at the origin
phi_ext = [phi_plat, phi_zeros, phi_zeros, phi_zeros]
# Name of the joint
name_Joint = ['Ankle', 'Knee', 'Hip']
# Euler sequences associated to each name_Joint
rot = [r2mobile.zyx, r2mobile.zxy, r2mobile.zxy]
name_rot = ['zyx', 'zxy', 'zxy']
# Point of calcul of the Moment and Force
point_limb = [segment_tibia.get_distal_frame_glob(),
              segment_thigh.get_distal_frame_glob(),
              segment_thigh.get_proximal_frame_glob()]
# Frame of expression of Moment and Force if not in JCS
frame_limb = [segment_tibia.Tdist, segment_thigh.Tdist, segment_pelvis.Tprox]
# Side
side = ['Right']

test_multiseg = KinematicChain(full_segment, phi_ext,
                               name_Joint, name_rot,
                               point_limb, frame_limb,
                               'JCS')
# Nomenclature ==> cohérence d'écriture lisibilité des variables

# ==============================================================================
# Validation
# ==============================================================================
# Kinetic Plot
# ------------------------------------------------------------------------------
ax = plt.subplot(331)
plt.plot(test_multiseg.moment['Ankle'][0, 0, :], label='PythonV2_autom')
plt.title('Ankle Mx')

plt.subplot(332, sharex=ax)
plt.plot(test_multiseg.moment['Ankle'][1, 0, :], label='PythonV2_autom')
plt.title('Ankle My')

plt.subplot(333, sharex=ax)
plt.plot(test_multiseg.moment['Ankle'][2, 0, :], label='PythonV2_autom')
plt.title('Ankle Mz')
plt.legend()

plt.subplot(334, sharex=ax)
plt.plot(test_multiseg.moment['Knee'][0, 0, :], label='PythonV2_autom')
plt.title('Knee Mx')

plt.subplot(335, sharex=ax)
plt.plot(test_multiseg.moment['Knee'][1, 0, :], label='PythonV2_autom')
plt.title('Knee My')

plt.subplot(336, sharex=ax)
plt.plot(test_multiseg.moment['Knee'][2, 0, :], label='PythonV2_autom')
plt.title('Knee Mz')
plt.legend()

plt.subplot(337, sharex=ax)
plt.plot(test_multiseg.moment['Hip'][0, 0, :], label='PythonV2_autom')
plt.title('Hip Mx')

plt.subplot(338, sharex=ax)
plt.plot(test_multiseg.moment['Hip'][1, 0, :], label='PythonV2_autom')
plt.title('Hip My')

plt.subplot(339, sharex=ax)
plt.plot(test_multiseg.moment['Hip'][2, 0, :], label='PythonV2_autom')
plt.title('Hip Mz')

plt.legend()
plt.show()

# Plot kinematics
# ------------------------------------------------------------------------------

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
plt.plot(test_multiseg.euler_rel['Hip'][0, :], label='Autom')
plt.title('Hip')

plt.subplot(332)
plt.plot(test_multiseg.euler_rel['Hip'][1, :], label='Autom')
plt.legend()
plt.title('Hip')

plt.subplot(333)
plt.plot(test_multiseg.euler_rel['Hip'][2, :], label='Autom')
plt.title('Hip')


plt.subplot(334)
plt.plot(test_multiseg.euler_rel['Knee'][0, :], label='Autom')
plt.title('Knee')


plt.subplot(335)
plt.plot(test_multiseg.euler_rel['Knee'][1, :], label='Autom')
plt.title('Knee')


plt.subplot(336)
plt.plot(test_multiseg.euler_rel['Knee'][2, :], label='Autom')
plt.title('Knee')


plt.subplot(337)
plt.plot(test_multiseg.euler_rel['Ankle'][0, :], label='Autom')
plt.title('Ankle')


plt.subplot(338)
plt.plot(test_multiseg.euler_rel['Ankle'][1, :], label='Autom')
plt.title('Ankle')


plt.subplot(339)
plt.plot(test_multiseg.euler_rel['Ankle'][2, :], label='Autom')
plt.title('Ankle')
plt.show()
