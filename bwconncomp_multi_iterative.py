import numpy as np
import math
import sys

import multiprocessing as mp

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
                'coord_end': (x_start + current_width -1, 
                              y_start + current_height -1),
                'label_range': (current_label, current_label + current_width * current_height -1),
            }
            blocks.append(block)
            
            y_start += current_height

            current_label += current_width * current_height
        
        x_start += current_width
    
    return blocks, blocks_x, blocks_y


def label_2d_block(BW, block, conn):
    x_start, y_start = block['coord_start']
    x_end, y_end = block['coord_end']

    labels = np.zeros(shape=(x_end-x_start +1, y_end-y_start +1))
    
    next_label = block['label_range'][0]

    equivalences = {}

    #FIRST PASS
    for x in range(x_start, x_end +1):
        for y in range(y_start, y_end +1):

            if BW[x, y] == 0:
                continue

            neighbors = []
            for dx, dy, dz in conn:
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
                        process_union(equivalences, n, min_label)

    #SECOND PASS
    for x in range(x_start, x_end +1):
        for y in range(y_start, y_end +1):
            if labels[x-x_start, y-y_start] != 0 and labels[x-x_start, y-y_start] in equivalences:
                labels[x-x_start, y-y_start] = find_root_label(equivalences, labels[x-x_start, y-y_start])

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

def process_union(equivalences, label1, label2):
    root1 = find_root_label(equivalences, label1)
    root2 = find_root_label(equivalences, label2)
    if root1 != root2:
        equivalences[max(root1, root2)] = min(root1, root2)

def find_root_label(equivalences, label):
    #find root label
    root = label
    while root in equivalences:
        root = equivalences[root]

    #path compression
    while label != root:
        parent = equivalences[label]
        equivalences[label] = root
        label = parent

    return root

#change to work with multiple cpus (multiprocessing)
def process_2d_equivalences(line_labels, conn):
    global_equivalences = {}

    for line in line_labels:
        first_half, second_half = line[0], line[1]

        search_extend = abs(1 - len(conn) // 4)

        for base in range(len(first_half)):
            for offset in range(0-search_extend, search_extend+1):
                extend = base + offset
                if extend < 0 or extend >= len(first_half):
                    continue

                if first_half[base] != 0 and second_half[extend] != 0:
                    process_union(global_equivalences, first_half[base], second_half[extend])

    return global_equivalences

def get_central_line_labels(blocks, num_rows, num_cols):

    central_line_labels = []

    #process vertical lines
    for j in range(num_cols-1):
        line_labels_left = []
        line_labels_right = []
        for i in range(num_rows):
            line_labels_left.extend(blocks[i][j]['border_right'])
            line_labels_right.extend(blocks[i][j+1]['border_left'])        
        
        vertical_line = [line_labels_left, line_labels_right]
        central_line_labels.append(vertical_line)

    #process horizontal lines
    for i in range(num_rows-1):
        line_labels_top = []
        line_labels_bottom = []
        for j in range(num_cols):
            line_labels_top.extend(blocks[i][j]['border_bottom'])
            line_labels_bottom.extend(blocks[i+1][j]['border_top'])    
        
        horizontal_line = [line_labels_top, line_labels_bottom]
        central_line_labels.append(horizontal_line)

    return central_line_labels



def merge_2d_blocks(BW, block, equivalances):

    x_start, y_start = block['coord_start']
    x_end, y_end = block['coord_end']

    for x in range(x_start, x_end +1):
        for y in range(y_start, y_end +1):
            label = block['labels'][x-x_start, y-y_start]
            if label != 0 and label in equivalances:
                label = find_root_label(equivalances, label)
            
            BW[x, y] = label


def bwconncomp_iterative(BW = None, conn = 4, cores = 1):
    """
    Accepts # cores that are a power of 2
    Careful with allocated cores
    Handles 2D images only

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
    pool = mp.Pool(cores)

    args = [(BW, block, M) for block in blocks]
    blocks = pool.starmap(label_2d_block, args)

    # for i, block in enumerate(blocks):
    #     blocks[i] = label_2d_block(BW, block, M)

    #change blocks to grid format:
    temp = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    for row in range(num_rows):
        for col in range(num_cols):
            temp[row][col] = blocks[row*num_cols + col]
    blocks = temp

    #figure out global equivalences for merging
    line_labels = get_central_line_labels(blocks, num_rows, num_cols)

    global_equivalences = process_2d_equivalences(line_labels, M)

    #merge blocks and resolve global equivalances
    new_BW = np.zeros(shape=(width, height))

    args = []

    for i in range(num_rows):
        for j in range(num_cols):
            block = blocks[i][j]
            args.append((new_BW, block, global_equivalences))
            
    pool.starmap(merge_2d_blocks, args)

    pool.close()

    # for i in range(num_rows):
    #     for j in range(num_cols):
    #         block = blocks[i][j]

    #         merge_2d_blocks(new_BW, block, global_equivalences)

    #sets up CC
    connectivity = conn
    imageSize = BW.shape
    numObjects = 0
    pixelIdxList = {}

    #change to work with multiple cpus (multiprocessing)
    for x in range(width):
        for y in range(height):
            pixel = BW[x, y]
            if pixel != 0:
                if pixel not in pixelIdxList:
                    pixelIdxList[pixel] = []
                    numObjects += 1
                
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

    # image, component_indices = gen_RNG_2dBW(50, 50, 15, 25, 50, conn4)

    # image, component_indices = gen_RNG_3dBW(50, 50, 50, 5, 25, 50, conn6)

    image, component_indices = BWTest.get_conn8_test(2)

    CC = bwconncomp_iterative(image, 8, 4)
    print(CC)

    Tester.test_bwconncomp_match(CC, image, component_indices)

    return 0

if __name__ == '__main__':
    if hasattr(sys, 'ps1'):  # Check if in interactive mode
        main()
    else:
        sys.exit(main())


