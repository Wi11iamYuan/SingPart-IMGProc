import numpy as np
import math
import sys

import multiprocessing as mp

import matplotlib.pyplot as plt
from constants import *

from bw_2d_gen import gen_RNG_2dBW



def regionprops(image, properties=None, output_format='struct'):
    """
    Accepts binary image (BW) or CC (connected components)

    Accepts properties as a list of strings, if none, all properties are calculated
    - Area, number of pixels in the region
    - Centroid, center of mass of the region, 
    - Bounding Box, smallest box containing the region

    Specify output format
    - Struct, array
    - Table, table

    """

    region_props = {}

    return region_props

def main():

    return 0


if __name__ == '__main__':
    if hasattr(sys, 'ps1'):  # Check if in interactive mode
        main()
    else:
        sys.exit(main())
