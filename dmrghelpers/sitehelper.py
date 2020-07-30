"""
创建sitebasis，符合常见的Hubbard模型的特点
"""

import numpy
from basics.basis import SiteBasis

#每个格子有4中状态，空o，向上u，向下d，满f
SITE_STATES = ['o', 'u', 'd', 'f']
#每种状态的粒子数，先上后下
STATE_PARTINUM = numpy.array([(0, 0), (1, 0), (0, 1), (1, 1)], dtype=numpy.int)


def site_of_idx(idx):
    '''在指定的idx上创建site'''
    stb = SiteBasis('s', [idx], SITE_STATES, STATE_PARTINUM)
    return stb
