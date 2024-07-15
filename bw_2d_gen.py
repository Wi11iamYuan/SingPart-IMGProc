import numpy as np
import random
import matplotlib.pyplot as plt
from .constants import *

def create_binary_image_with_components(width:int=150, height:int=150, max_comp:int=5, max_comp_pix:int=500, max_attempts=100, conn = conn4):
    image = np.zeros((width, height), dtype=int)
    
    for i in range(1, max_comp + 1):
        attempts = 0
        while attempts < max_attempts:
            x, y = random.randint(0, width-1), random.randint(0, height-1)
            if image[x, y] == 0:
                grow_component(image, x, y, i, max_comp_pix, conn)
                break
            attempts += 1
        if attempts == max_attempts:
            print(f"Warning: Could not place component {i}")
    
    return image

def grow_component(image, x, y, component_label, max_pixels, conn):
    stack = [(x, y, 0)]
    pixels_filled = 0
    while stack and pixels_filled < max_pixels:
        x, y, z = stack.pop()
        if 0 <= x < image.shape[0] and 0 <= y < image.shape[1] and image[x, y] == 0:
            image[x, y] = component_label
            pixels_filled += 1
            neighbors = [(x + dx, y + dy, dz) for dx, dy, dz in conn]
            
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if random.randint(1, 10) <= 7:
                    stack.append(neighbor) 


def get_component_indices(image, max_comp:int=5):
    components = {}
    for i in range(1, max_comp + 1):
        component_indices = np.where(image == i)
        if len(component_indices[0]) > 0:
            components[i] = list(component_indices[0] * image.shape[1] + component_indices[1])
    return components

# Create the image
image = create_binary_image_with_components(150,150,13,500,1000,conn4)

# Get indices for each component
component_indices = get_component_indices(image, 13)

# Print the indices for each component
for component, indices in component_indices.items():
    print(f"Component {component}: {indices}")

# Display the image
plt.imshow(image)