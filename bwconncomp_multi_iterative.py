import numpy as np
import math
import sys

import multiprocessing
import threading

import matplotlib.pyplot as plt
import pyvista as pv
from constants import *

from img_conversion import img_toBW
from bw_2d_gen import gen_RNG_2dBW
from bw_3d_gen import gen_RNG_3dBW

def generate_2d_blocks(BW, cores):
    width = BW.shape[0]
    height = BW.shape[1]

    blocks_x = 1
    blocks_y = 1

    power = (int)(math.log2(cores))
    x_turn = True

    for i in range(power):
        if x_turn:
            blocks_x *= 2
        else:
            blocks_y *= 2
        x_turn = not x_turn
    
    # Calculate block sizes
    block_width = width // blocks_x
    block_height = height // blocks_y

    #note the remainder pixels
    remainder_x = width % blocks_x
    remainder_y = height % blocks_y

    #help debug
    # print(blocks_x, blocks_y)
    # print(block_width, block_height)
    # print(remainder_x, remainder_y)

    blocks = []

    current_label = 1
    y_start = 0
    for i in range(blocks_y):
        # Adjust block height if there are remainder pixels to distribute
        current_height = block_height + (1 if i < remainder_y else 0)
        x_start = 0
        
        for j in range(blocks_x):
            # Adjust block width if there are remainder pixels to distribute
            current_width = block_width + (1 if j < remainder_x else 0)
            
            # Create a block with its coordinates and size
            block = {
                'coord_start': (x_start, y_start),
                'coord_end': (x_start + current_width -1, 
                              y_start + current_height -1),
                'label_range': (current_label, current_label + current_width * current_height -1),
            }
            blocks.append(block)
            
            x_start += current_width

            current_label += current_width * current_height
        
        y_start += current_height
    
    return blocks, blocks_x, blocks_y

def label_2d_blocks(BW, block, conn):
    x_start, y_start = block['coord_start']
    x_end, y_end = block['coord_end']

    labels = np.zeros(shape=(x_end-x_start +1, y_end-y_start +1))
    
    start_label = block['label_range'][0]
    next_label = block['label_range'][0]

    equivalences = {}

    #FIRST PASS
    for x in range(x_start, x_end +1):
        for y in range(y_start, y_end +1):

            if BW[x, y] == 0:
                continue

            neighbors = []
            for dx, dy in conn:
                nx, ny = x + dx, y + dy

                if x_start <= nx <= x_end and y_start <= ny <= y_end:
                    neighbors.append(labels[nx-x_start, ny-y_start])


            if not neighbors or all(n==0 for n in neighbors):
                labels[x-x_start, y-y_start] = next_label
                next_label += 1
            else:
                min_label = min(n for n in neighbors if n != 0)
                labels[x-x_start, y-y_start] = min_label

                for n in neighbors:
                    if n != 0 and n != min_label:
                        equivalences[n] = min_label

    #RESOLVE EQUIVALENCES
    for label in range(start_label, next_label):
        if label in equivalences:
            root = label
            while root in equivalences:
                root = equivalences[root]
            equivalences[label] = root

    #SECOND PASS
    for x in range(x_start, x_end +1):
        for y in range(y_start, y_end +1):
            if labels[x-x_start, y-y_start] != 0:
                labels[x-x_start, y-y_start] = equivalences[labels[x-x_start, y-y_start]]

    new_block = {
        'coord_start': (x_start, y_start),
        'coord_end': (x_end, y_end),
        'labels': labels,
        'border_top': labels[0, :],
        'border_bottom': labels[-1, :],
        'border_left': labels[:, 0],
        'border_right': labels[:, -1],
    }

    return new_block

#change to work with multiple cpus (multiprocessing)
def process_2d_equivalences(blocks, num_rows, num_cols):
    global_equivalences = {}

    #process horizontal merges
    for i in range(num_rows -1):
        for j in range(num_cols):
            process_horizontal(blocks[i][j], blocks[i+1][j], global_equivalences)
    
    #process vertical merges
    for i in range(num_rows):
        for j in range(num_cols -1):
            process_vertical(blocks[i][j], blocks[i][j+1], global_equivalences)

    return global_equivalences


def process_horizontal(upper_block, lower_block, global_equivalences):
    upper_border = upper_block['border_bottom']
    lower_border = lower_block['border_top']

    for col in range(upper_border.shape[0]):
        if upper_border[col] != 0 and lower_border[col] != 0:
            process_union(global_equivalences, upper_border[col], lower_border[col])

def process_vertical(left_block, right_block, global_equivalences):
    left_border = left_block['border_right']
    right_border = right_block['border_left']

    for row in range(left_border.shape[0]):
        if left_border[row] != 0 and right_border[row] != 0:
            process_union(global_equivalences, left_border[row], right_border[row])


def process_union(global_equivalences, label1, label2):
    root1 = find_root_label(global_equivalences, label1)
    root2 = find_root_label(global_equivalences, label2)
    if root1 != root2:
        global_equivalences[max(root1, root2)] = min(root1, root2)


def find_root_label(global_equivalences, label):
    root = label
    while root in global_equivalences:
        root = global_equivalences[root]

    #path compression
    while label != root:
        parent = global_equivalences[label]
        global_equivalences[label] = root
        label = parent

    return root

#change to work with multiple cpus (multiprocessing)
def merge_2d_blocks(blocks, equivalances, width, height):
    new_BW = np.zeros(shape=(width, height))

    for block in blocks:
        x_start, y_start = block['coord_start']
        x_end, y_end = block['coord_end']

        for x in range(x_start, x_end +1):
            for y in range(y_start, y_end +1):
                label = block['labels'][x-x_start, y-y_start]
                if label != 0 and label in equivalances:
                    label = equivalances[label]
                
                new_BW[x, y] = label

    return new_BW


def bwconncomp_iterative(BW = None, conn: int | None = None, cores: int | None = None):
    """
    Accepts # cores that are a power of 2

    """

    if not ((cores & (cores-1) == 0) and cores > 0):
        raise ValueError("Number of cores must be a power of 2")
    
    #turns BW into a numpy array
    BW = np.asarray(BW)
    width = BW.shape[0]
    height = BW.shape[1]

    blocks = []

    #checks if the image is 2D or 3D, then assigns the correct connectivity
    M = None
    if BW.ndim == 2:
        if conn == 4:
            M = conn4
        else: # conn = 8
            M = conn8
        blocks, num_rows, num_cols = generate_2d_blocks(BW, cores)
    else: # BW.ndim == 3
        raise NotImplementedError("3D not implemented yet")
        # if conn == 6:
        #     M = conn6
        # elif conn == 18:
        #     M = conn18
        # else: # conn = 26
        #     M = conn26

    #change to work with multiple cpus (multiprocessing)
    for i, block in enumerate(blocks):
        blocks[i] = label_2d_blocks(BW, block, M)

    #change blocks to grid format:
    temp = [num_rows][num_cols]
    for row in range(num_rows):
        for col in range(num_cols):
            temp[row][col] = blocks[row*num_cols + col]
    
    blocks = temp

    #figure out global equivalences for merging
    global_equivalences = process_2d_equivalences(blocks, num_rows, num_cols)

    #merge blocks and resolve global equivalances
    BW = merge_2d_blocks(blocks, global_equivalences, width, height)

    #sets up CC
    connectivity = conn
    imageSize = BW.shape
    numObjects = 0
    pixelIdxList = {}


    for x in range(width):
        for y in range(height):
            pixel = BW[x, y]
            if pixel != 0:
                if pixel not in pixelIdxList:
                    pixelIdxList[pixel] = []
                
                pixelIdxList[pixel].append(x * height + y)

    pixelIdxList = list(pixelIdxList.values())

    #creates the CC
    CC = {
        'Connectivity': connectivity,
        'ImageSize': imageSize,
        'NumObjects': numObjects,
        'PixelIdxList': pixelIdxList
    }
                
    return CC
    
def main():

    # image, component_indices = gen_RNG_2dBW(50, 50, 5, 25, 50, conn4)

    # image, component_indices = gen_RNG_3dBW(50, 50, 50, 5, 25, 50, conn6)

    # image, component_indices = BWTest.get_conn26_test(0)

    # CC = bwconncomp_iterative(image, 26)

    # Tester.test_bwconncomp_match(CC, image, component_indices)

    # image = np.zeros((1423, 1443))
    # blocks, x, y = generate_2d_blocks(image, cores=8)

    # # Print block information
    # for i, block in enumerate(blocks):
    #     print(f"Block {i}: From ({block['coord_start']} "
    #         f"to ({block['coord_end']}), with range {block['label_range']}")



    return 0

if __name__ == '__main__':
    if hasattr(sys, 'ps1'):  # Check if in interactive mode
        main()
    else:
        sys.exit(main())


