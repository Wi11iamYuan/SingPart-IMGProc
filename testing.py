from bwconncomp_base_recursive import bwconncomp_recursive
from bwconncomp_multi_iterative import bwconncomp_iterative
from constants import *

import sys


def main():
    # image, component_indices = BWTest.get_conn4_test(1)
    image = Tester.process_large_tests("./large_tests/2d_1000x1000_BW_0.txt")

    print("Iterative")
    Tester.test_bwconncomp_time(image, 8, bwconncomp_iterative)

    print("Recursive")
    Tester.test_bwconncomp_time(image, 8, bwconncomp_recursive)

    return 0

if __name__ == '__main__':
    if hasattr(sys, 'ps1'):  # Check if in interactive mode
        main()
    else:
        sys.exit(main())
