import numpy as np


def vnop_array(V, e1, e2, e3):
    vnop = np.zeros((3, V.shape[1]))

    vnop[0, :] = np.sum(np.cross(e2, e3, axis=0) * V, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)
    vnop[1, :] = np.sum(np.cross(e3, e1, axis=0) * V, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)
    vnop[2, :] = np.sum(np.cross(e1, e2, axis=0) * V, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)

    return vnop
