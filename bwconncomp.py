import numpy as np
import math
import sys

import multiprocessing as mp

import matplotlib.pyplot as plt
from constants import *

from bw_2d_gen import gen_RNG_2dBW


def generate_2d_blocks(BW, cores):
    #define for later use
    width = BW.shape[0]
    height = BW.shape[1]

    blocks_x = 1
    blocks_y = 1

    #how many times to split the image
    power = (int)(math.log2(cores))
    x_turn = True

    #splits the image alternating between x and y
    for i in range(power):
        if x_turn:
            blocks_x *= 2
        else:
            blocks_y *= 2
        x_turn = not x_turn
    
    #calculate block sizes
    block_width = width // blocks_x
    block_height = height // blocks_y

    #note the remainder pixels
    remainder_x = width % blocks_x
    remainder_y = height % blocks_y

    blocks = []

    current_label = 1
    x_start = 0
    for i in range(blocks_x):
        #adjust block width if there are remainder pixels to distribute
        current_width = block_width + (1 if i < remainder_x else 0)
        y_start = 0
        
        for j in range(blocks_y):
            #adjust block height if there are remainder pixels to distribute
            current_height = block_height + (1 if j < remainder_y else 0)
            
            #create a block with its coordinates and size
            block = {
                'coord_start': (x_start, y_start),
                'coord_end': (x_start + current_width -1, 
                              y_start + current_height -1),
                'label_range': (current_label, current_label + current_width * current_height -1),
            }
            blocks.append(block)

            #update coordinates
            y_start += current_height

            current_label += current_width * current_height
        
        x_start += current_width
    
    return blocks, blocks_x, blocks_y


def label_2d_block(BW, block, conn):
    #define for later use
    x_start, y_start = block['coord_start']
    x_end, y_end = block['coord_end']

    #blank labels to fill in
    labels = np.zeros(shape=(x_end-x_start +1, y_end-y_start +1))
    
    next_label = block['label_range'][0]

    equivalences = {}

    #FIRST PASS
    for x in range(x_start, x_end +1):
        for y in range(y_start, y_end +1):

            if BW[x, y] == 0:
                continue

            #if not zero, add the pixels to neighbors from labels
            neighbors = []
            for dx, dy, dz in conn:
                nx, ny = x + dx, y + dy

                if x_start <= nx <= x_end and y_start <= ny <= y_end:
                    neighbors.append(labels[nx-x_start, ny-y_start])

            #if no neighbors or other pixels haven't been labeled yet, assign a new label
            if not neighbors or all(n==0 for n in neighbors):
                labels[x-x_start, y-y_start] = next_label
                next_label += 1
            else:
                #find the minimum label to give the pixel
                min_label = min(n for n in neighbors if n != 0)
                labels[x-x_start, y-y_start] = min_label

                #process equivalences and resolve unions
                for n in neighbors:
                    if n != 0 and n != min_label:
                        process_union(equivalences, n, min_label)

    #SECOND PASS
    for x in range(x_start, x_end +1):
        for y in range(y_start, y_end +1):
            #resolves equivalences
            if labels[x-x_start, y-y_start] != 0 and labels[x-x_start, y-y_start] in equivalences:
                labels[x-x_start, y-y_start] = find_root_label(equivalences, labels[x-x_start, y-y_start])

    #update block with labels and borders
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
    #creates a chain between the two labels in equivalences for later processing
    root1 = find_root_label(equivalences, label1)
    root2 = find_root_label(equivalences, label2)
    if root1 != root2:
        equivalences[max(root1, root2)] = min(root1, root2)

def find_root_label(equivalences, label):
    #find root label to find true component
    root = label
    while root in equivalences:
        root = equivalences[root]

    #path compression, changes others to root so is faster in later runs
    while label != root:
        parent = equivalences[label]
        equivalences[label] = root
        label = parent

    return root

def process_2d_equivalences(line, conn, equivalences):

    #gets line info
    first_half, second_half = line[0], line[1]

    #search extend is the maximum distance to search for equivalences, works for 4 and 8 connectivity
    search_extend = abs(1 - len(conn) // 4)

    for base in range(len(first_half)):
        #offset can be 0 for conn4, or -1, 0, 1 for conn8
        for offset in range(0-search_extend, search_extend+1):
            
            extend = base + offset
            if extend < 0 or extend >= len(first_half):
                continue

            #if both labels are not zero, process union, since they connect
            if first_half[base] != 0 and second_half[extend] != 0:
                process_union(equivalences, first_half[base], second_half[extend])

def get_central_line_labels(blocks, num_rows, num_cols):

    central_line_labels = []

    #process vertical lines
    for j in range(num_cols-1):
        line_labels_left = []
        line_labels_right = []
        for i in range(num_rows):
            #appends the border information
            line_labels_left.extend(blocks[i][j]['border_right'])
            line_labels_right.extend(blocks[i][j+1]['border_left'])        
        
        vertical_line = [line_labels_left, line_labels_right]
        central_line_labels.append(vertical_line)

    #process horizontal lines
    for i in range(num_rows-1):
        line_labels_top = []
        line_labels_bottom = []
        for j in range(num_cols):
            #appends the border information
            line_labels_top.extend(blocks[i][j]['border_bottom'])
            line_labels_bottom.extend(blocks[i+1][j]['border_top'])    
        
        horizontal_line = [line_labels_top, line_labels_bottom]
        central_line_labels.append(horizontal_line)

    return central_line_labels



def merge_2d_blocks(BW, block, equivalances):

    #coords for the block
    x_start, y_start = block['coord_start']
    x_end, y_end = block['coord_end']

    for x in range(x_start, x_end +1):
        for y in range(y_start, y_end +1):
            #finds the label and checks if it is in the equivalences
            label = block['labels'][x-x_start, y-y_start]
            if label != 0 and label in equivalances:
                label = find_root_label(equivalances, label)
            
            #assigns the label to the new image officially
            BW[x, y] = label



def bwconncomp(BW = None, conn = 4, cores = 1):
    """
    Accepts # cores that are a power of 2
    Careful with allocated cores
    Handles 2D images only
    """
    #does bit manipulation to check if cores is a power of 2
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

    #uses multiprocessing
    pool = mp.Pool(cores)

    #label blocks
    args = [(BW, block, M) for block in blocks]
    blocks = pool.starmap(label_2d_block, args)

    # for i, block in enumerate(blocks):
    #     blocks[i] = label_2d_block(BW, block, M)

    #change blocks to grid format:
    grid_blocks = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    for row in range(num_rows):
        for col in range(num_cols):
            grid_blocks[row][col] = blocks[row*num_cols + col]

    #get central line labels for merging, they are the central lines that run through the image to create the blocks
    line_labels = get_central_line_labels(grid_blocks, num_rows, num_cols)

    #figure out global equivalences for merging
    global_equivalences = {}
    for line in line_labels:
        process_2d_equivalences(line, M, global_equivalences)

    #merge blocks and resolve global equivalances
    new_BW = np.zeros(shape=(width, height))

    args = []

    for block in blocks:
        args.append((new_BW, block, global_equivalences))
            
    #merge the blocks
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

    #creates the pixelIdxList
    for x in range(width):
        for y in range(height):
            pixel = BW[x, y]
            if pixel != 0:
                #if new component, add to the list
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

    CC = bwconncomp(image, 8, 4)
    print(CC)

    Tester.test_bwconncomp_match(CC, image, component_indices)

    return 0

if __name__ == '__main__':
    if hasattr(sys, 'ps1'):  # Check if in interactive mode
        main()
    else:
        sys.exit(main())


