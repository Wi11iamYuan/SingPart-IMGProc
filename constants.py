"""
Support for 2D and 3D binary images

Width x Height x Depth
X x Y x Z

Ignore PNG alpha channel
0.5 threshold = 128 int

Sort components based on top-left extremum with row-major order
1. Highest z (for 3D)
2. Highest y
3. Lowest x
"""


conn4 = [
    [0, 1, 0],   # Up
    [-1, 0, 0],  # Left
    [1, 0, 0],   # Right
    [0, -1, 0],  # Down
]

conn8 = [
    [-1, 1, 0],   # Top-left
    [0, 1, 0],    # Top
    [1, 1, 0],    # Top-right
    [-1, 0, 0],   # Left
    [1, 0, 0],    # Right
    [-1, -1, 0],  # Bottom-left
    [0, -1, 0],   # Bottom
    [1, -1, 0],    # Bottom-right
]

conn6 = [
    [0, 0, 1],    # Top (higher z)
    [0, 1, 0],    # Back (higher y)
    [-1, 0, 0],   # Left (lower x)
    [1, 0, 0],    # Right
    [0, -1, 0],   # Front (lower y)
    [0, 0, -1],    # Bottom (lower z)
]

conn18 = [
    [0, 1, 1],    # Top-back
    [-1, 0, 1],   # Top-left
    [0, 0, 1],    # Top
    [1, 0, 1],    # Top-right
    [0, -1, 1],   # Top-front
    [-1, 1, 0],   # Back-left
    [0, 1, 0],    # Back
    [1, 1, 0],    # Back-right
    [-1, 0, 0],   # Left
    [1, 0, 0],    # Right
    [-1, -1, 0],  # Front-left
    [0, -1, 0],   # Front
    [1, -1, 0],   # Front-right
    [0, 1, -1],   # Bottom-back
    [-1, 0, -1],  # Bottom-left
    [0, 0, -1],   # Bottom
    [1, 0, -1],   # Bottom-right
    [0, -1, -1],  # Bottom-front
]

conn26 = [
    [-1, 1, 1],   # Top-back-left
    [0, 1, 1],    # Top-back
    [1, 1, 1],    # Top-back-right
    [-1, 0, 1],   # Top-left
    [0, 0, 1],    # Top
    [1, 0, 1],    # Top-right
    [-1, -1, 1],  # Top-front-left
    [0, -1, 1],   # Top-front
    [1, -1, 1],   # Top-front-right
    [-1, 1, 0],   # Back-left
    [0, 1, 0],    # Back
    [1, 1, 0],    # Back-right
    [-1, 0, 0],   # Left
    [1, 0, 0],    # Right
    [-1, -1, 0],  # Front-left
    [0, -1, 0],   # Front
    [1, -1, 0],   # Front-right
    [-1, 1, -1],  # Bottom-back-left
    [0, 1, -1],   # Bottom-back
    [1, 1, -1],   # Bottom-back-right
    [-1, 0, -1],  # Bottom-left
    [0, 0, -1],   # Bottom
    [1, 0, -1],   # Bottom-right
    [-1, -1, -1], # Bottom-front-left
    [0, -1, -1],  # Bottom-front
    [1, -1, -1]   # Bottom-front-right
]