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
    if x < 0 or x >= BW.shape[0] or y < 0 or y >= BW.shape[1] or BW[x, y] == 0 or TRACK[x, y] == 1:
        return
    
    idx = x * BW.shape[1] + y
    idxList.append(idx)
    TRACK[x, y] = 1

    for offset in M:
        nx, ny = x + offset[0], y + offset[1]
        bwconncomp_flood2d(BW, TRACK, M, idxList, nx, ny)

def bwconncomp_flood3d(BW, TRACK, M, idxList, x, y, z):
    if x < 0 or x >= BW.shape[0] or y < 0 or y >= BW.shape[1] or z < 0 or z >= BW.shape[2] or BW[x, y, z] == 0 or TRACK[x, y, z] == 1:
        return
    
    idx = x * BW.shape[1] * BW.shape[2] + y * BW.shape[2] + z
    idxList.append(idx)
    TRACK[x, y, z] = 1

    for offset in M:
        nx, ny, nz = x + offset[0], y + offset[1], z + offset[2]
        bwconncomp_flood3d(BW, TRACK, M, idxList, nx, ny, nz)

#%%
def bwconncomp(BW = None, conn: int | None = None):
    BW = np.asarray(BW)
    TRACK = np.zeros(BW.shape, dtype=int)

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

    connectivity = conn
    imageSize = BW.shape
    numObjects = 0
    pixelIdxList = []

    if BW.ndim == 2:
        for i in range(BW.shape[0]):
            for j in range(BW.shape[1]):
                if BW[i, j] > 0 and TRACK[i, j] == 0:
                    numObjects += 1
                    idxList = []
                    bwconncomp_flood2d(BW, TRACK, M, idxList, i, j)
                    pixelIdxList.append(idxList)
    else: # BW.ndim == 3
        for i in range(BW.shape[0]):
            for j in range(BW.shape[1]):
                for k in range(BW.shape[2]):
                    if BW[i, j, k] > 0 and TRACK[i, j, k] == 0:
                        numObjects += 1
                        idxList = []
                        bwconncomp_flood3d(BW, TRACK, M, idxList, i, j, k)
                        pixelIdxList.append(idxList)

    pixelIdxList = [sorted(x) for x in pixelIdxList]

    CC = {
        'Connectivity': connectivity,
        'ImageSize': imageSize,
        'NumObjects': numObjects,
        'PixelIdxList': pixelIdxList
    }
                
    return CC
    
# %%
def main():

    # test_2d, test_2d_indices = gen_RNG_2dBW(50, 50, 5, 25, 50, conn4)

    # CC = bwconncomp(test_2d, 4)

    # plt.imshow(test_2d)

    # print(f"Objects detected: {CC['NumObjects']}")

    # for i in range(0, CC["NumObjects"]):
    #     print(f"Component {i+1} Matches") if CC['PixelIdxList'][i] == test_2d_indices[i] else print(f"Component {i+1} Does Not Match")
    #     print("bwconncomp")
    #     print(CC['PixelIdxList'][i])
    #     print("test")
    #     print(test_3d_indices[i])

    test_3d, test_3d_indices = gen_RNG_3dBW(50, 50, 50, 5, 25, 50, conn6)

    CC = bwconncomp(test_3d, 6)

    print(f"Objects detected: {CC['NumObjects']}")

    for i in range(0, CC["NumObjects"]):
        print(f"Component {i+1} Matches") if CC['PixelIdxList'][i] == test_3d_indices[i] else print(f"Component {i+1} Does Not Match")
        print("bwconncomp")
        print(CC['PixelIdxList'][i])
        print("test")
        print(test_3d_indices[i])

    plt = pv.Plotter()
    plt.add_volume(test_3d, cmap="viridis")
    plt.show()

    return 0

if __name__ == '__main__':
    if hasattr(sys, 'ps1'):  # Check if in interactive mode
        main()
    else:
        sys.exit(main())


# %%
