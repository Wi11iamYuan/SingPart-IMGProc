import numpy as np
import random

import matplotlib.pyplot as plt
from constants import *

import sys

def create_binary_image_with_components(width:int=150, height:int=150, max_comp:int=5, max_comp_pix:int=500, max_attempts=100, conn=conn4):
    image = np.zeros((width, height), dtype=int)
    
    for i in range(1, max_comp + 1):
        attempts = 0
        while attempts < max_attempts:
            x, y = random.randint(0, width-1), random.randint(0, height-1)
            if image[x, y] == 0 and not has_adjacent_component(image, x, y, conn):
                grow_component(image, x, y, i, max_comp_pix, conn)
                break
            attempts += 1
        if attempts == max_attempts:
            print(f"Warning: Could not place component {i}")
    
    return image

def grow_component(image, x, y, component_label, max_pixels, conn):
    stack = [(x, y)]
    pixels_filled = 0
    while stack and pixels_filled < max_pixels:
        x, y = stack.pop()
        if 0 <= x < image.shape[0] and 0 <= y < image.shape[1] and image[x, y] == 0:
            image[x, y] = component_label
            pixels_filled += 1
            neighbors = [(x + dx, y + dy) for dx, dy, _ in conn]
            
            random.shuffle(neighbors)
            for nx, ny in neighbors:
                if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and image[nx, ny] == 0:
                    if not has_adjacent_component(image, nx, ny, conn, exclude_label=component_label):
                        if random.randint(1, 10) <= 6:
                            stack.append((nx, ny))

def has_adjacent_component(image, x, y, conn, exclude_label=None):
    for dx, dy, _ in conn:
        nx, ny = x + dx, y + dy
        if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
            if image[nx, ny] != 0 and image[nx, ny] != exclude_label:
                return True
    return False

def get_component_indices(image, max_comp:int=5):
    components = []
    for i in range(1, max_comp + 1):
        component_indices = np.where(image == i)
        if len(component_indices[0]) > 0:
            components.append(sorted(list(component_indices[0] * image.shape[1] + component_indices[1])))

    components = sorted(components, key=lambda item: (item[0]))
    return components

def gen_RNG_2dBW(x,y,max_comp,max_comp_pix,max_attempts,conn):
    BW = create_binary_image_with_components(x,y,max_comp,max_comp_pix,max_attempts,conn)

    component_indices = get_component_indices(BW, max_comp)

    return BW, component_indices



def main():
    # Create the image
    image = create_binary_image_with_components(150,150,13,500,1000,conn4)

    # Get indices for each component
    component_indices = get_component_indices(image, 13)

    # Print the indices for each component
    for i in range(len(component_indices)):
        print(f"Component {i+1}: {component_indices[i]}")

    # Display the image
    plt.imshow(image)

    return 0

if __name__ == '__main__':
    if hasattr(sys, 'ps1'):  # Check if in interactive mode
        main()
    else:
        sys.exit(main())