from bwconncomp_base_recursive import bwconncomp_recursive
from bwconncomp_multi_iterative import bwconncomp_iterative
from constants import *

import sys


def main():
    image, component_indices = BWTest.get_conn4_test(1)
    
    print("Recursive")
    Tester.test_bwconncomp_time(image, 4, bwconncomp_recursive)

    print("Iterative")
    Tester.test_bwconncomp_time(image, 4, bwconncomp_iterative)

    return 0

if __name__ == '__main__':
    if hasattr(sys, 'ps1'):  # Check if in interactive mode
        main()
    else:
        sys.exit(main())
