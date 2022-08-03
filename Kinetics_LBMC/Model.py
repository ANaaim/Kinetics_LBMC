import treelib
from .multi_body_optimisation import multi_body_optimisation as multi_body_optimisation
from .KinematicChain import KinematicChain as KinematicChain
import numpy as np
import time


class Model:
    def __init__(self, list_joint_mbo, list_joint_kinematics_only=None):
        """Initiation of the model based on two list. The joint that will be used in the multi body optimisation
        represent the different link for the multiple body joint optimisation. The other link such as link from a segment
        to a forceplateforme shoud be put in the list_joint_kinematics_only

        :param list_joint_mbo: list of _description_
        :type list_joint_mbo: _type_
        :param list_joint_kinematics_only: _description_, defaults to None
        :type list_joint_kinematics_only: _type_, optional
        """
        # list_joint_mbo is the list of joints of the multibody optimisation
        # To be able to add other solid that should not be optimised such as
        # the forceplatform, we need to add them only when we do calculation of the kinematics
        self.list_joint_mbo = list_joint_mbo
        self.list_joint_kinematics_only = list_joint_kinematics_only
        # from a list of joint, create a list of segment
        joined_list = list_joint_mbo + list_joint_kinematics_only
        list_child = list()
        all_segment = list()
        for joint in joined_list:
            if joint.segment_distal not in list_child:
                list_child.append(joint.segment_distal)

            if joint.segment_proximal not in all_segment:
                all_segment.append(joint.segment_proximal)
            if joint.segment_distal not in all_segment:
                all_segment.append(joint.segment_distal)
        list_root = list()
        for segment in all_segment:
            if segment not in list_child:
                list_root.append(segment)

        self.list_root = list_root

    def mbo(self, max_iter=50, time_process=False):
        """_summary_

        :param max_iter: number of iteration allowed during the newton-Raphson optimisation, defaults to 50
        :type max_iter: int, optional
        :param time_process: Define if the multibody optmisation should be time, defaults to False
        :type time_process: bool, optional
        """

        list_full_segment, list_joint, list_link = create_segment_list_from_joint(self.list_joint_mbo)
        if time_process:
            start_time = time.time()
        multi_body_optimisation(list_full_segment, list_joint, max_iter)
        if time_process:
            final_time = time.time() - start_time
            print("--- %s seconds ---" % (final_time))
        # Nothing shoudl be returned as the data are modified in the multibody optimisation function
        return

    def kinematics_only(self, point_frq, gravity_direction, unit_point, cut_off_frequency=12, projection_moment="JCS"):
        joined_list_joint = self.list_joint_mbo + self.list_joint_kinematics_only

        list_full_segment, list_joint, list_link = create_segment_list_from_joint(joined_list_joint)
        full_list_all_path = list()
        full_list_all_path_joint = list()
        for root in self.list_root:
            list_all_path, list_all_path_joint = create_data_for_kinematics(list_full_segment, root, list_link)
            full_list_all_path += list_all_path
            full_list_all_path_joint += list_all_path_joint

        all_multiseg = calculate_multiseg(
            joined_list_joint,
            list_full_segment,
            full_list_all_path,
            full_list_all_path_joint,
            point_frq,
            cut_off_frequency,
            gravity_direction,
            unit_point,
            projection_moment,
        )
        return all_multiseg


def create_segment_list_from_joint(list_joint):
    # How to create a model from model_joint only

    # From the list of joint we can construct a model
    # First step :
    #   -Add all segment to a list : list_full_segment
    #  - Extract all the joint all the link between the segment as their position in the list :list_link
    #  - add in the joint
    list_full_segment = list()
    list_link = list()
    for joint in list_joint:
        # updata of the list of segment in the model
        if joint.segment_distal not in list_full_segment:
            list_full_segment.append(joint.segment_distal)
        if joint.segment_proximal not in list_full_segment:
            list_full_segment.append(joint.segment_proximal)
        position_segment_proximal_joint = list_full_segment.index(joint.segment_proximal)
        position_segment_distal_joint = list_full_segment.index(joint.segment_distal)
        list_link.append((position_segment_proximal_joint, position_segment_distal_joint))
        # Modify the information of the position of the distal and proximal segment in the joint
        # TODO this information could be given by the user ????
        joint.proximal_indice = position_segment_proximal_joint
        joint.distal_indice = position_segment_distal_joint

    return list_full_segment, list_joint, list_link


def create_data_for_kinematics(list_full_segment, root, list_link):

    pos_of_root = list_full_segment.index(root)
    connexion = np.zeros((len(list_full_segment), len(list_full_segment)))
    for (X, Y) in list_link:
        connexion[X, Y] = 1
        connexion[Y, X] = 1
    link_to_add = True

    test_tree = treelib.Tree()
    test_tree.create_node(list_full_segment[pos_of_root].segment_name, pos_of_root)
    node_to_analyse = [pos_of_root]
    while link_to_add:
        futur_node_to_analyse = list()
        for node in node_to_analyse:
            for pos, value in enumerate(connexion[node, :]):
                if value == 1:
                    futur_node_to_analyse.append(pos)
                    test_tree.create_node(list_full_segment[pos].segment_name, pos, parent=node)
                    # we remove the link
                    connexion[pos, node] = 0
                    connexion[node, pos] = 0
        if len(futur_node_to_analyse) == 0:
            link_to_add = False
        else:
            node_to_analyse = futur_node_to_analyse
            print(node_to_analyse)
    # Extract all the path to leave
    list_all_path = test_tree.paths_to_leaves()
    list_all_path_joint = list()
    for list_path in list_all_path:
        list_joint_temp = list()
        for ind in range(len(list_path) - 1):
            if (list_path[ind], (list_path[ind + 1])) in list_link:
                list_joint_temp.append(list_link.index((list_path[ind], (list_path[ind + 1]))))
            elif (list_path[ind + 1], (list_path[ind])) in list_link:
                list_joint_temp.append(list_link.index((list_path[ind + 1], (list_path[ind]))))
        list_all_path_joint.append(list_joint_temp)

    return list_all_path, list_all_path_joint


def calculate_multiseg(
    list_full_joint,
    list_full_segment,
    list_all_path,
    list_all_path_joint,
    point_frq,
    cut_off_frequency,
    gravity_direction,
    unit_point,
    projection_moment,
):
    # Creation of the information for the kinematics chain from the link
    list_all_segment = list()
    list_all_phi_ext = list()
    list_all_name_joint = list()
    list_all_name_rotation = list()
    list_all_pos_moment_calculation = list()
    list_all_frame_moment_calculation = list()

    for list_segment, list_joint in zip(list_all_path, list_all_path_joint):
        # In order to construct the date for the kinematics chain we need
        # to reverse the list of the segment
        segment = list()
        phi_ext = list()
        name_joint = list()
        name_rotation = list()
        pos_moment_calculation = list()
        frame_moment_calculation = list()
        for position_segment in list_segment[::-1]:
            segment_to_consider = list_full_segment[position_segment]
            segment.append(segment_to_consider)
            phi_ext.append(segment_to_consider.phi_ext)

        # In order to construct the data for the kinematics chain we need
        # to reverse the list of the segmen
        for position_joint in list_joint[::-1]:
            joint_to_consider = list_full_joint[position_joint]
            name_joint.append(joint_to_consider.name_joint)
            name_rotation.append(joint_to_consider.euler_sequences)
            pos_moment_calculation.append(joint_to_consider.point_of_moment_calculus)
            frame_moment_calculation.append(joint_to_consider.frame_moment)

        list_all_segment.append(segment)
        list_all_phi_ext.append(phi_ext)
        list_all_name_joint.append(name_joint)
        list_all_name_rotation.append(name_rotation)
        list_all_pos_moment_calculation.append(pos_moment_calculation)
        list_all_frame_moment_calculation.append(frame_moment_calculation)

    all_multiseg = list()
    for ind_chain in range(len(list_all_segment)):
        list_segment = list_all_segment[ind_chain]
        list_phi_ext = list_all_phi_ext[ind_chain]
        list_name_joint = list_all_name_joint[ind_chain]
        list_name_rotation = list_all_name_rotation[ind_chain]
        list_pos_moment_calculation = list_all_pos_moment_calculation[ind_chain]
        list_frame_moment_calculation = list_all_frame_moment_calculation[ind_chain]

        multiseg = KinematicChain(
            list_segment,
            list_phi_ext,
            list_name_joint,
            list_name_rotation,
            list_pos_moment_calculation,
            list_frame_moment_calculation,
            point_frq,
            cut_off_frequency,
            gravity_direction,
            unit_point,
            projection_moment,
        )
        all_multiseg.append(multiseg)

    return all_multiseg
