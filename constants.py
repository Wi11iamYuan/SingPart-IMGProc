"""
Support for 2D and 3D binary images

Width x Height x Depth
X x Y x Z

Ignore PNG alpha channel
0.5 threshold = 128 int

Sort components based on smallest index

"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

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

class BWTest():

    def __init__(self):
        pass

    conn4_test1_BW = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 4, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 4, 4, 0, 0, 4, 4, 0, 0, 0, 0, 1, 1, 1, 1, 0],
    [4, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
    ]  

    conn4_test1_components = [[28, 48, 68, 86, 87, 88, 105, 106, 107, 108, 126, 127, 128, 129, 130, 131, 150, 151, 153, 170, 171, 172, 173, 191, 192], [40], [205, 206, 226, 227, 228, 229, 249, 263, 264, 265, 269, 280, 281, 282, 283, 285, 286, 289, 290, 300, 306, 307, 308, 309, 310], [220, 240], [295, 296, 297, 298, 314, 315, 316, 317, 333, 334, 335, 336, 337, 353, 354, 355, 356, 357, 358, 359, 373, 379, 393, 398, 399]]

    conn4_test2_BW = [
    [ 0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9,  0,  0,  0],
    [ 5,  5,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0],
    [ 5,  5,  5,  0,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
    [ 5,  5,  0,  0,  2,  0,  7,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0],
    [ 0,  5,  0,  2,  2,  0,  0,  4,  4,  0,  0,  0,  0,  1,  1,  0,  0,  6,  6,  0],
    [ 5,  5,  0,  2,  2,  2,  2,  0,  0,  1,  1,  1,  1,  1,  1,  0,  0,  6,  6,  0],
    [ 5,  5,  0,  0,  0,  2,  2,  0,  0,  1,  0,  0,  0,  0,  1,  0, 10,  0,  0,  0],
    [ 0,  5,  0,  0,  0,  2,  0,  0,  0,  0,  0,  3,  3,  0,  0, 10, 10, 10,  0,  0],
    [ 0,  5,  0,  0,  0,  2,  0,  0,  8,  8,  0,  3,  3,  3,  0, 10, 10, 10,  0, 10],
    [ 0,  5,  0,  0,  0,  0,  8,  8,  8,  8,  0,  3,  3,  3,  0, 10, 10, 10, 10, 10]
    ]

    conn4_test2_components = [[5, 25, 44, 45, 64, 83, 84, 103, 104, 105, 106, 125, 126, 145, 165], [16], [20, 21, 40, 41, 42, 60, 61, 81, 100, 101, 120, 121, 141, 161, 181], [34, 35, 54, 73, 74, 93, 94, 109, 110, 111, 112, 113, 114, 129, 134], [66], [87, 88], [97, 98, 117, 118], [136, 155, 156, 157, 175, 176, 177, 179, 195, 196, 197, 198, 199], [151, 152, 171, 172, 173, 191, 192, 193], [168, 169, 186, 187, 188, 189]]

    conn4_test3_BW = [
    [0, 0, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0],
    [0, 0, 0, 6, 6, 0, 0, 6, 6, 0, 0, 0],
    [3, 3, 3, 0, 6, 0, 0, 6, 0, 2, 2, 0],
    [3, 0, 3, 0, 0, 0, 0, 0, 0, 2, 2, 0],
    [3, 3, 3, 3, 0, 0, 0, 2, 2, 0, 2, 2],
    [3, 0, 3, 0, 1, 1, 0, 2, 2, 2, 2, 2],
    [3, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 5, 5, 5, 5],
    [0, 0, 0, 0, 1, 1, 0, 7, 0, 0, 0, 5],
    [0, 0, 0, 1, 1, 1, 0, 0, 4, 4, 4, 0]
    ]

    conn4_test3_components = [[2, 3, 4, 5, 6, 7, 8, 15, 16, 19, 20, 28, 31], [24, 25, 26, 36, 38, 48, 49, 50, 51, 60, 62, 72, 73], [33, 34, 45, 46, 55, 56, 58, 59, 67, 68, 69, 70, 71], [64, 65, 75, 76, 77, 87, 88, 89, 100, 101, 111, 112, 113], [92, 93, 94, 95, 107], [103], [116, 117, 118]]

    conn4_tests = [(conn4_test1_BW, conn4_test1_components), (conn4_test2_BW, conn4_test2_components), (conn4_test3_BW, conn4_test3_components)]

    @staticmethod
    def get_conn4_test(idx):
        return BWTest.conn4_tests[idx]
    
    conn8_test1_BW = [
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [3, 0, 1, 0, 0],
    [0, 0, 1, 0, 2],
    [0, 1, 0, 0, 2]
    ]

    conn8_test1_components = [[7, 12, 17, 21], [10], [19, 24]]

    conn8_test2_BW = [
    [2, 2, 2, 0, 0, 0, 6, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 2, 2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [2, 2, 0, 2, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 1],
    [2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 1, 1, 0, 1, 1],
    [2, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 1, 1, 1, 1, 1],
    [0, 0, 3, 3, 3, 0, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1],
    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 5, 5, 5, 5, 5, 5],
    [4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 0, 5, 5, 5, 5, 5, 5],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 5, 5, 5, 5, 5, 5]
    ]

    conn8_test2_components = [[0, 1, 2, 17, 18, 20, 21, 34, 35, 37, 38, 39, 51, 52, 53, 54, 55, 68, 69, 70, 71, 85, 86, 102], [6, 7], [10, 26, 28, 42, 43, 46, 47, 49, 65, 67, 84, 97, 98, 100, 101, 114, 115, 116, 117, 118, 132, 133, 134, 135], [77, 78, 91, 92, 93, 94, 95, 105, 106, 107, 108, 109, 110, 111, 112, 121, 122, 123, 125, 126, 127, 128, 129, 137], [158, 159, 160, 161, 162, 170, 171, 173, 174, 175, 176, 177, 178, 179, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196], [164, 165, 166, 167, 168, 169, 181, 182, 183, 184, 185, 186, 198, 199, 200, 201, 202, 203]]

    conn8_test3_BW = [
    [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 4],
    [0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 4],
    [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 2]
    ]
    
    conn8_test3_components = [[1, 2, 3, 8, 9, 10, 14, 17, 19, 20, 21, 22, 23, 27, 30, 31, 33, 34, 35, 36, 46, 47, 48, 49], [12, 25], [51]]

    conn8_tests = [(conn8_test1_BW, conn8_test1_components), (conn8_test2_BW, conn8_test2_components), (conn8_test3_BW, conn8_test3_components)]

    @staticmethod
    def get_conn8_test(idx):
        return BWTest.conn8_tests[idx]
    
    conn6_test1_BW = [
    [[2, 2, 0], [2, 2, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[2, 2, 2], [2, 2, 0], [2, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1]],
    [[2, 2, 2], [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 0, 1], [1, 1, 1]],
    [[2, 2, 2], [2, 2, 2], [0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 0, 0]]
    ]

    conn6_test1_components = [[0, 1, 3, 4, 18, 19, 20, 21, 22, 24, 36, 37, 38, 39, 40, 41, 42, 43, 44, 54, 55, 56, 57, 58, 59], [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 28, 30, 31, 32, 34, 35, 48, 50, 51, 52, 53, 64, 65, 67, 68]]
    
    conn6_test2_BW = [
    [[0, 3, 3, 0, 2, 2], [3, 3, 3, 0, 2, 2], [3, 3, 3, 0, 2, 2], [3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0]],
    [[3, 3, 3, 0, 2, 2], [3, 3, 3, 0, 2, 2], [3, 3, 3, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 4]],
    [[3, 3, 3, 0, 0, 0], [3, 3, 3, 0, 0, 0], [3, 0, 0, 0, 6, 6], [0, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 4], [0, 1, 0, 1, 0, 4]],
    [[0, 0, 0, 5, 5, 5], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 4], [1, 1, 0, 0, 4, 4], [0, 1, 0, 1, 0, 4]],
    [[5, 5, 5, 5, 5, 5], [5, 5, 5, 0, 0, 0], [5, 5, 0, 4, 4, 0], [0, 0, 0, 0, 4, 4], [1, 1, 1, 0, 4, 4], [0, 1, 1, 1, 0, 4]],
    [[5, 5, 5, 5, 0, 0], [5, 5, 5, 0, 4, 4], [5, 5, 5, 0, 4, 4], [0, 0, 0, 0, 4, 4], [1, 1, 1, 0, 4, 4], [1, 1, 0, 0, 4, 4]]
    ]

    conn6_test2_components = [[1, 2, 6, 7, 8, 12, 13, 14, 18, 36, 37, 38, 42, 43, 44, 48, 49, 50, 72, 73, 74, 78, 79, 80, 84], [4, 5, 10, 11, 16, 17, 40, 41, 46, 47], [28, 29, 57, 58, 64, 93, 97, 99, 103, 105, 132, 133, 139, 141, 168, 169, 170, 175, 176, 177, 204, 205, 206, 210, 211], [71, 101, 107, 130, 131, 136, 137, 143, 159, 160, 166, 167, 172, 173, 179, 190, 191, 196, 197, 202, 203, 208, 209, 214, 215], [88, 89], [111, 112, 113, 144, 145, 146, 147, 148, 149, 150, 151, 152, 156, 157, 180, 181, 182, 183, 186, 187, 188, 192, 193, 194]]

    conn6_test3_BW = [
    [[2, 2], [0, 0], [1, 1], [1, 1]],
    [[2, 2], [0, 0], [1, 1], [1, 1]],
    [[0, 0], [1, 1], [1, 1], [0, 1]],
    [[3, 3], [0, 0], [0, 0], [0, 0]],
    [[0, 3], [3, 3], [3, 3], [3, 3]]
    ]

    conn6_test3_components = [[0, 1, 8, 9], [4, 5, 6, 7, 12, 13, 14, 15, 18, 19, 20, 21, 23], [24, 25, 33, 34, 35, 36, 37, 38, 39]]

    conn6_tests = [(conn6_test1_BW, conn6_test1_components), (conn6_test2_BW, conn6_test2_components), (conn6_test3_BW, conn6_test3_components)]

    @staticmethod
    def get_conn6_test(idx):
        return BWTest.conn6_tests[idx]
    
    conn18_test1_BW = [
    [[0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 1, 0, 0, 1, 0, 2], [3, 0, 0, 0, 1, 0, 0, 2], [3, 0, 3, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 1, 1, 0, 0, 2], [3, 0, 1, 1, 1, 0, 0, 2], [3, 0, 0, 1, 0, 1, 0, 2], [3, 3, 0, 0, 0, 0, 0, 2]],
    [[0, 0, 1, 0, 0, 0, 2, 2], [3, 0, 0, 0, 1, 0, 2, 2], [3, 3, 0, 0, 1, 0, 0, 2], [3, 3, 3, 0, 0, 0, 0, 2]]
    ]

    conn18_test1_components = [[7, 15, 23, 39, 47, 55, 63, 70, 71, 78, 79, 87, 95], [10, 13, 20, 35, 36, 42, 43, 44, 51, 53, 66, 76, 84], [16, 24, 26, 40, 48, 56, 57, 72, 80, 81, 88, 89, 90]]

    conn18_test2_BW = [
    [[0, 3, 3, 0, 0], [0, 0, 0, 0, 2], [0, 2, 2, 2, 2], [0, 2, 0, 2, 0]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 2], [2, 2, 0, 2, 0]],
    [[0, 1, 1, 0, 0], [0, 1, 0, 0, 2], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
    [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 0, 0, 2], [0, 1, 0, 0, 2]],
    [[1, 1, 0, 4, 4], [1, 1, 0, 0, 0], [1, 0, 0, 2, 0], [1, 0, 0, 0, 2]]
    ]

    conn18_test2_components = [[1, 2], [9, 11, 12, 13, 14, 16, 18, 28, 34, 35, 36, 38, 49, 74, 79, 93, 99], [41, 42, 46, 52, 60, 61, 65, 66, 67, 71, 76, 80, 81, 85, 86, 90, 95], [83, 84]]

    conn18_test3_BW = [
    [[2, 2, 2, 2, 2], [0, 0, 2, 2, 2], [0, 0, 0, 0, 2], [0, 1, 1, 0, 0]],
    [[2, 2, 2, 2, 2], [0, 0, 2, 0, 2], [1, 0, 0, 0, 0], [0, 1, 0, 1, 0]],
    [[0, 2, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 0, 0]],
    [[0, 0, 0, 0, 0], [1, 0, 0, 1, 0], [1, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
    [[0, 0, 0, 0, 3], [0, 0, 1, 0, 0], [1, 1, 0, 0, 4], [0, 0, 0, 0, 0]]
    ]

    conn18_test3_components = [[0, 1, 2, 3, 4, 7, 8, 9, 14, 20, 21, 22, 23, 24, 27, 29, 41], [16, 17, 30, 36, 38, 50, 51, 53, 57, 65, 68, 70, 72, 78, 87, 90, 91], [84], [94]]

    conn18_tests = [(conn18_test1_BW, conn18_test1_components), (conn18_test2_BW, conn18_test2_components), (conn18_test3_BW, conn18_test3_components)]

    @staticmethod
    def get_conn18_test(idx):
        return BWTest.conn18_tests[idx]
    
    conn26_test1_BW = [
    [[2, 0, 0, 1, 1], [2, 0, 1, 1, 1]],
    [[2, 0, 1, 1, 1], [2, 0, 1, 1, 1]],
    [[2, 0, 1, 1, 1], [2, 0, 0, 1, 1]],
    [[2, 0, 0, 0, 0], [2, 0, 0, 1, 0]],
    [[2, 2, 0, 0, 0], [2, 2, 0, 0, 0]]
    ]

    conn26_test1_components = [[0, 5, 10, 15, 20, 25, 30, 35, 40, 41, 45, 46], [3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24, 28, 29, 38]]

    conn26_test2_BW = [
    [[0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], [2, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]],
    [[0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 1, 1, 0, 1, 1], [2, 2, 2, 0, 2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 1, 0]]
    ]

    conn26_test2_components = [[1, 2, 16, 19, 33, 34, 35, 36, 38, 39, 48, 49, 50, 52, 53, 54, 55], [10, 11, 12, 13, 14, 25, 26, 27, 31, 43, 44, 46, 47, 58, 59, 61, 62]]

    conn26_test3_BW = [
    [[5, 0, 0, 0, 0], [0, 0, 2, 0, 2], [0, 0, 0, 2, 0], [0, 1, 0, 2, 0], [1, 0, 0, 2, 2], [1, 0, 0, 0, 0]],
    [[0, 0, 0, 2, 0], [0, 0, 0, 2, 0], [1, 0, 0, 0, 0], [1, 0, 0, 2, 2], [1, 1, 0, 2, 0], [1, 0, 0, 0, 2]],
    [[0, 0, 0, 0, 2], [0, 1, 0, 2, 2], [1, 0, 0, 2, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 2], [1, 0, 0, 0, 0], [1, 1, 0, 6, 6]],
    [[0, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 6, 6]]
    ]

    conn26_test3_components = [[0], [7, 9, 13, 18, 23, 24, 33, 38, 48, 49, 53, 59, 64, 68, 69, 73, 109], [16, 20, 25, 40, 45, 50, 51, 55, 66, 70, 75, 76, 85, 110, 115, 116, 145], [118, 119, 148, 149], [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 137]]

    conn26_tests = [(conn26_test1_BW, conn26_test1_components), (conn26_test2_BW, conn26_test2_components), (conn26_test3_BW, conn26_test3_components)]

    @staticmethod
    def get_conn26_test(idx):
        return BWTest.conn26_tests[idx]


class Tester():

    def __init__(self):
        pass

    @staticmethod
    def test_bwconncomp_match(CC, image, component_indices):
        print(f"Objects detected: {CC['NumObjects']}")

        for i in range(0, CC["NumObjects"]):
            print(f"Component {i+1} Matches") if CC['PixelIdxList'][i] == component_indices[i] else print(f"Component {i+1} Does Not Match")
            print("bwconncomp")
            print(CC['PixelIdxList'][i])
            print("test")
            print(component_indices[i])

        if len(CC['ImageSize']) == 3:
            vol_img = np.array(image)
            vol = pv.Plotter()
            vol.add_volume(vol_img, cmap="viridis")
            vol.show()
        else:
            plt.imshow(image)




