import argparse
import os
import sys
import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt

def quantize(v, palette):
    """
    Given a scalar v and array of values palette,
    return the index of the closest value
    """

    if np.isscalar(v):

        index = np.argmin(np.abs(v - palette))
        return index
    else:
        if v.ndim == 1:
            v = v[:, np.newaxis]

        indices = np.argmin(np.abs(v - palette), axis=-1)
        return indices



# def quantize(v, palette):
#     """
#     Given a scalar v and array of values palette,
#     return the index of the closest value
#     """
#     # ~~START DELETE~~
#     # Quantizing multiple
#     return np.argmin(np.abs(v.reshape(-1, 1) - palette.reshape(1, -1)), axis=1)
#     # ~~END DELETE~~
#     return 0

# Example usage:
v_scalar = np.array([10])
palette = np.array([0, 10, 20, 30])
print("Scalar quantize result:", quantize(v_scalar, palette))

v_vector = np.array([5, 15, 25])
print("Vector quantize result:", quantize(v_vector, palette))