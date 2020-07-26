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
        template += 'Basis: %s\n' % self._basis.__class__.__name__
        for idx in range(self._basis.dim):
            template += '%.4f |%s>\n' %\
                (self._comps[idx], self._basis.idx_to_state(idx))
        template += 'dim: %d' % self._basis.dim
        return template

    @property
    def components(self):
        '''各个分量'''
        return self._comps

class SparseStates(object):
    """
    一个比较稀疏的态，利用dict保存状态而不是一个列向量
    """
    def __init__(self, bss: Basis, initdic: dict):
        '''initdic中应该是（int, float）的对'''
        self._basis = bss
        self._comps_dic = initdic

    def __str__(self):
        template = 'SparseStates:\n'
        template += 'Basis: %s\n' % self._basis.__class__.__name__
        for key in self._comps_dic:
            template += '%4f |%s>\n' % \
                (self._comps_dic[key], self._basis.idx_to_state(key))
        return template

    @property
    def components(self):
        '''返回态的字典'''
        comps = {}
        for key in self._comps_dic:
            comps[key] = self._comps_dic[key]
        return comps

    def to_dense(self):
        '''转换成稠密的表示'''
        initv = numpy.zeros([self._basis.dim, 1])
        for idx in self._comps_dic:
            initv[idx, 0] = self._comps_dic[idx]
        return States(self._basis, initv)
