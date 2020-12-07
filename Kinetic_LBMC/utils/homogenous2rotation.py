"""Calculate rotation matrix from the proximal and distal homogenous matrix."""


def homogenous2rotation(homo_proximal, homo_distal):
    """Calculate rotation matrix from the proximal and distal homogenous matrix.

    Transformation from an homogenous of the distal segment and the proximal
    segment to the rotation matrix from the distal to the proximal

    :param homo_proximal: Homogenous matrix of the proximal segment
    :type homo_proximal: mat_Homo
    :param homo_distal: Homogenous matrix of the distal segment
    :type homo_distal: mat_Homo
    :return: Homogenous
    :rtype: mat_Homo

    """
    rotation_mat = homo_proximal.inv() * homo_distal

    return rotation_mat
