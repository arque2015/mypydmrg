"""整理profile文件的内容"""

import sys
import pstats


def main(argv):
    '''开始程序'''
    ofile = open(argv[2] if len(argv) > 2 else argv[1] + '.out', 'w')
    prof = pstats.Stats(argv[1], stream=ofile)
    prof.sort_stats('time')
    prof.print_stats()


if __name__ == "__main__":
    main(sys.argv)
