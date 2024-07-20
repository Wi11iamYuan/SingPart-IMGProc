#%%
import numpy as np
from math import ceil, floor
import sys

import multiprocessing
import threading

import matplotlib.pyplot as plt
import pyvista as pv
from constants import *

from img_conversion import img_toBW
from bw_2d_gen import gen_RNG_2dBW
from bw_3d_gen import gen_RNG_3dBW

#%%
def generate_2d_blocks(BW, cores):
    width = BW.shape[0]
    height = BW.shape[1]

    aspect_ratio = width / height
    
    # Calculate blocks_x and blocks_y
    blocks_x = ceil(np.sqrt(cores * aspect_ratio))
    blocks_y = floor(cores / blocks_x)
    
    # Adjust if the total is less than cores
    while blocks_x * blocks_y < cores:
        if aspect_ratio > 1:
            blocks_x += 1
        else:
            blocks_y += 1
    
    # Calculate block sizes
    block_width = ceil(width / blocks_x)
    block_height = ceil(height / blocks_y)

    #note the remainder pixels
    remainder_x = width % blocks_x
    remainder_y = height % blocks_y

    blocks = []

    current_label = 1
    x_start = 0
    for i in range(blocks_x):
        # Adjust block width if there are remainder pixels to distribute
        current_width = block_width + (1 if i < remainder_x else 0)
        y_start = 0
        
        for j in range(blocks_y):
            # Adjust block height if there are remainder pixels to distribute
            current_height = block_height + (1 if j < remainder_y else 0)
            
            # Create a block with its coordinates and size
            block = {
                'coord_start': (x_start, y_start),
                'coord_end': (x_start + current_width, 
                              y_start + current_height),
                'label_range': (current_label, current_label + current_width * current_height-1),
            }
            blocks.append(block)
            
            y_start += current_height

            current_label += current_width * current_height
        
        x_start += current_width
    
    return blocks

def generate_3d_blocks(BW, cores):
    width = BW.shape[0]
    height = BW.shape[1]
    depth = BW.shape[2]

    # Calculate the ratios
    ratio_yx = height / width
    ratio_zx = depth / width

    # Calculate the number of blocks in each dimension
    blocks_x = ceil((cores / (ratio_yx * ratio_zx)) ** (1/3))
    blocks_y = ceil(blocks_x * ratio_yx)
    blocks_z = ceil(blocks_x * ratio_zx)

    # Calculate the size of each block
    block_width = width // blocks_x
    block_height = height // blocks_y
    block_depth = depth // blocks_z

    # Note the remainder pixels
    remainder_x = width % blocks_x
    remainder_y = height % blocks_y
    remainder_z = depth % blocks_z

    blocks = []

    x_start = 0
    for i in range(blocks_x):
        # Adjust block width if there are remainder pixels to distribute
        current_width = block_width + (1 if i < remainder_x else 0)
        y_start = 0
        
        for j in range(blocks_y):
            # Adjust block height if there are remainder pixels to distribute
            current_height = block_height + (1 if j < remainder_y else 0)
            z_start = 0

            for k in range(blocks_z):
                # Adjust block depth if there are remainder pixels to distribute
                current_depth = block_depth + (1 if k < remainder_z else 0)
            
                # Create a block with its coordinates and size
                block = {
                    'coord_start': (x_start, y_start, z_start),
                    'coord_end': (x_start + current_width, 
                                  y_start + current_height, 
                                  z_start + current_depth),
                    'label_range': (),
                }
                blocks.append(block)

                z_start += current_depth
            
            y_start += current_height
        
        x_start += current_width

    return blocks


#%%
def bwconncomp_iterative(BW = None, conn: int | None = None, cores: int | None = None):
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

    # image, component_indices = BWTest.get_conn26_test(0)

    # CC = bwconncomp_iterative(image, 26)

    # Tester.test_bwconncomp_match(CC, image, component_indices)

    # Example usage:
    image = np.zeros((1456, 1520))  # Example image of size 1000x1500
    blocks = generate_2d_blocks(image, cores=16)

    # Print block information
    for i, block in enumerate(blocks):
        print(f"Block {i}: From ({block['coord_start']} "
            f"to ({block['coord_end']}), wit hrange {block['label_range']}")

    return 0

if __name__ == '__main__':
    if hasattr(sys, 'ps1'):  # Check if in interactive mode
        main()
    else:
        sys.exit(main())


# %%
