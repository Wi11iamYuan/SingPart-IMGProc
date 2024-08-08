from archive.bwconncomp_base_recursive import bwconncomp_recursive
from bwconncomp import bwconncomp
from constants import *

import sys


def main():
    # image, component_indices = BWTest.get_conn4_test(1)
    image = Tester.process_large_tests("./large_tests/2d_2048x2048_BW_8_0.txt")

    # cores = 8
    # print("Iterative")
    # print("Cores: ", cores)
    # Tester.test_bwconncomp_time(image, 8, bwconncomp, cores)

    # print("Recursive")
    # Tester.test_bwconncomp_time(image, 8, bwconncomp_recursive)

    # Tester.run_bwconncomp_analysis(image, 8, bwconncomp_recursive, cores = 1, times = 20)
    # Tester.run_bwconncomp_analysis(image, 8, bwconncomp, cores = 1, times = 20)
    # Tester.run_bwconncomp_analysis(image, 8, bwconncomp, cores = 2, times = 20)
    
    Tester.run_multicore_bwconncomp_analysis(image, 8, bwconncomp, 32, 20)

    return 0

if __name__ == '__main__':
    if hasattr(sys, 'ps1'):  # Check if in interactive mode
        main()
    else:
        sys.exit(main())
