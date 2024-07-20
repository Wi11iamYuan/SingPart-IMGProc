#%%
import numpy as np

import multiprocessing
import threading

import matplotlib.pyplot as plt
import pyvista as pv
from constants import *

import sys

from img_conversion import img_toBW
from bw_2d_gen import gen_RNG_2dBW
from bw_3d_gen import gen_RNG_3dBW

#%%
def bwconncomp_iterative(BW = None, conn: int | None = None):
    #turns BW into a numpy array, then creates an empty track for the recursion later
    BW = np.asarray(BW)

    #checks if the image is 2D or 3D, then assigns the correct connectivity
    M = None
    if BW.ndim == 2:
        if conn == 4:
            M = conn4
        else: # conn = 8
            M = conn8
    else: # BW.ndim == 3
        if conn == 6:
            M = conn6
        elif conn == 18:
            M = conn18
        else: # conn = 26
            M = conn26

    #sets up CC
    connectivity = conn
    imageSize = BW.shape
    numObjects = 0
    pixelIdxList = []

    #creates the CC
    CC = {
        'Connectivity': connectivity,
        'ImageSize': imageSize,
        'NumObjects': numObjects,
        'PixelIdxList': pixelIdxList
    }
                
    return CC
    
# %%
def main():

    # image, component_indices = gen_RNG_2dBW(50, 50, 5, 25, 50, conn4)

    # image, component_indices = gen_RNG_3dBW(50, 50, 50, 5, 25, 50, conn6)

    image, component_indices = BWTest.get_conn26_test(0)

    CC = bwconncomp_iterative(image, 26)

    Tester.test_bwconncomp_match(CC, image, component_indices)

    return 0

if __name__ == '__main__':
    if hasattr(sys, 'ps1'):  # Check if in interactive mode
        main()
    else:
        sys.exit(main())


# %%
