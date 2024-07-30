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

def process_2d_equivalences(blocks, num_rows, num_cols, conn):
    global_equivalences = {}

    #process horizontal merges
    for i in range(num_rows -1):
        for j in range(num_cols):
            process_horizontal(blocks[i][j], blocks[i+1][j], global_equivalences, conn)
    
    #process vertical merges
    for i in range(num_rows):
        for j in range(num_cols -1):
            process_vertical(blocks[i][j], blocks[i][j+1], global_equivalences, conn)

    return global_equivalences

def process_horizontal(upper_block, lower_block, global_equivalences, conn):
    upper_border = upper_block['border_bottom']
    lower_border = lower_block['border_top']

    search_extend = abs(1 - len(conn) // 4)

    for base in range(len(upper_border)):
        for offset in range (0-search_extend, search_extend+1):
            extend = base + offset
            if extend < 0 or extend >= len(lower_border):
                continue

            if upper_border[base] != 0 and lower_border[extend] != 0:
                process_union(global_equivalences, upper_border[base], lower_border[extend])

def process_vertical(left_block, right_block, global_equivalences, conn):
    left_border = left_block['border_right']
    right_border = right_block['border_left']

    search_extend = abs(1 - len(conn) // 4)

    for base in range(len(left_border)):
        for offset in range (0-search_extend, search_extend+1):
            extend = base + offset
            if extend < 0 or extend >= len(right_border):
                continue

            if left_border[base] != 0 and right_border[extend] != 0:
                process_union(global_equivalences, left_border[base], right_border[extend])

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