#%%
import numpy as np

import multiprocessing
import threading
import imageio.v3 as iio

from constants import *
from img_conversion import img_toBW

#%%
'''
Consider doing this but iteratively for larger images
Consider sorting the components like bwconncomp
Consider combining the two functions somehow
'''
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
                if BW[i, j] == 1 and TRACK[i, j] == 0:
                    numObjects += 1
                    idxList = []
                    bwconncomp_flood2d(BW, TRACK, M, idxList, i, j)
                    pixelIdxList.append(idxList)
    else: # BW.ndim == 3
        for i in range(BW.shape[0]):
            for j in range(BW.shape[1]):
                for k in range(BW.shape[2]):
                    if BW[i, j, k] == 1 and TRACK[i, j, k] == 0:
                        numObjects += 1
                        idxList = []
                        bwconncomp_flood3d(BW, TRACK, M, idxList, i, j, k)
                        pixelIdxList.append(idxList)

    CC = {
        'Connectivity': connectivity,
        'ImageSize': imageSize,
        'NumObjects': numObjects,
        'PixelIdxList': pixelIdxList
    }
                
    return CC
    
# %%
BW2d = [
   [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
   [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
   [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [1, 0, 0, 1, 1, 1, 0, 1, 0, 0],
   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
]

BW2d_bool = [
   [bool(x) for x in row] for row in BW2d
]

BW3d = [
    [
        [1, 0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 1]
    ],
    [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1],
        [1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 1, 0]
    ],
    [
        [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0]
    ],
    [
        [0, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0]
    ]
]

# CC = bwconncomp(BW2d_bool, 8)
# print(CC["Connectivity"])
# print(CC["ImageSize"])
# print(CC["NumObjects"])
# print(CC["PixelIdxList"])

# CC = bwconncomp(BW3d, 18)
# print(CC["Connectivity"])
# print(CC["ImageSize"])
# print(CC["NumObjects"])
# print(CC["PixelIdxList"])

TEST_BW = img_toBW("./imgs/test.png")
CC = bwconncomp(TEST_BW, 8)
print(CC["Connectivity"])
print(CC["ImageSize"])
print(CC["NumObjects"])
print(CC["PixelIdxList"])
# %%
for i in CC["PixelIdxList"]:
    print(i[0])
# %%
