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
def bwconncomp_flood2d(BW, TRACK, M, idxList, x, y):
    #if the pixel is out of bounds, background, or already tracked, return
    if x < 0 or x >= BW.shape[0] or y < 0 or y >= BW.shape[1] or BW[x, y] == 0 or TRACK[x, y] == 1:
        return
    
    #add the pixel to the idxList and mark it as tracked
    idx = x * BW.shape[1] + y
    idxList.append(idx)
    TRACK[x, y] = 1

    #recursively call the function on the pixel's neighbors
    for offset in M:
        nx, ny = x + offset[0], y + offset[1]
        bwconncomp_flood2d(BW, TRACK, M, idxList, nx, ny)

def bwconncomp_flood3d(BW, TRACK, M, idxList, x, y, z):
    #if the pixel is out of bounds, background, or already tracked, return
    if x < 0 or x >= BW.shape[0] or y < 0 or y >= BW.shape[1] or z < 0 or z >= BW.shape[2] or BW[x, y, z] == 0 or TRACK[x, y, z] == 1:
        return
    
    #add the pixel to the idxList and mark it as tracked
    idx = x * BW.shape[1] * BW.shape[2] + y * BW.shape[2] + z
    idxList.append(idx)
    TRACK[x, y, z] = 1

    #recursively call the function on the pixel's neighbors
    for offset in M:
        nx, ny, nz = x + offset[0], y + offset[1], z + offset[2]
        bwconncomp_flood3d(BW, TRACK, M, idxList, nx, ny, nz)

#%%
def bwconncomp(BW = None, conn: int | None = None):
    #turns BW into a numpy array, then creates an empty track for the recursion later
    BW = np.asarray(BW)
    TRACK = np.zeros(BW.shape, dtype=int)

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

    if BW.ndim == 2:
        #searches through every pixel in the image
        for i in range(BW.shape[0]):
            for j in range(BW.shape[1]):
                #if the pixel is not background and has not been tracked yet, start a new component
                if BW[i, j] > 0 and TRACK[i, j] == 0:
                    numObjects += 1
                    idxList = []
                    #recursively flood-fill the component, adding pixels to the idxList
                    bwconncomp_flood2d(BW, TRACK, M, idxList, i, j)
                    pixelIdxList.append(idxList)
    else: # BW.ndim == 3
        #searches through every pixel in the image
        for i in range(BW.shape[0]):
            for j in range(BW.shape[1]):
                for k in range(BW.shape[2]):
                    #if the pixel is not background and has not been tracked yet, start a new component
                    if BW[i, j, k] > 0 and TRACK[i, j, k] == 0:
                        numObjects += 1
                        idxList = []
                        #recursively flood-fill the component, adding pixels to the idxList
                        bwconncomp_flood3d(BW, TRACK, M, idxList, i, j, k)
                        pixelIdxList.append(idxList)

    #sorts the idxLists for each list to be in ascending order
    pixelIdxList = [sorted(x) for x in pixelIdxList]

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

    CC = bwconncomp(image, 26)

    Tester.test_bwconncomp_match(CC, image, component_indices)

    return 0

if __name__ == '__main__':
    if hasattr(sys, 'ps1'):  # Check if in interactive mode
        main()
    else:
        sys.exit(main())


# %%
