"""
和量子态有关的功能
"""

import numpy
from .basis import Basis

class States(object):
    """
    一个态，|S> S属于Basis.States
    """
    def __init__(
            self,
            bss: Basis,#在什么基上
            initv,#初始化的值，如果是数组，那么是各个分量，如果是整数数字，就是这个分量
        ):
        self._basis = bss
        self._comps = numpy.zeros([bss.dim, 1])#initv
        if isinstance(initv, int):
            self._comps[initv] = 1.
        else:
            if len(initv.shape) < 2:
                initv = numpy.expand_dims(initv, 1)
            self._comps += initv

    def __str__(self):
        template = 'States:\n'
        for val, bss in zip(self._comps, self._basis.states):
            template += '%.4f |%s>\n' % (val[0], bss)
        template += 'dim: %d' % self._basis.dim
        return template

    @property
    def components(self):
        '''各个分量'''
        return self._comps
