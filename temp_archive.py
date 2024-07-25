from math import ceil, floor

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
