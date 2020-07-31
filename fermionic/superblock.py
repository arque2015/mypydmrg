"""
superblock有关的功能
computational many particle physics （21.23） （21.8）
"""

import numpy
from collections import namedtuple
from basics.basis import ProdBasis
from .block import LeftBlockExtend, RightBlockExtend
from . import DEBUG_MODE

class SuperBlockExtend(ProdBasis):
    """将一个LeftBlockExtend和一个RightBlockExtend
    直积成一个superblock，这时为了方便，直接拆成4个，如果想要得到每个idx
    使用ProdBasis.idx_to_sitecode或者self.idx_to_idxtuple
    如果向和LeftBlockExtend中的idx对应，可以使用LeftBlockExtend中的函数
    RightBlockExtend同理
    ``````
    注意之前的LeftBlockExtend都是LeftBlock + 单个site（Right也一样）
    """
    def __init__(self, lbke: LeftBlockExtend, rbke: RightBlockExtend):
        self._lbke = lbke
        self._rbke = rbke
        self._lblk = lbke.lblk
        self._lsite = lbke.stbss
        self._rsite = rbke.stbss
        self._rblk = rbke.rblk
        #
        stadic = {}
        #注意SiteBasis和Block的prefix没有做特别的区分
        #在这里作出一些区分
        stadic['l'+self._lblk.prefix] = self._lblk.states
        #SiteBasis的states是单个site的所有状态
        stadic['l'+self._lsite.prefix] = self._lsite.states
        stadic['r'+self._rsite.prefix] = self._rsite.states
        #
        stadic['r'+self._rblk.prefix] = self._rblk.states
        #
        super().__init__(stadic)
        #
        self._block_len = lbke.block_len + rbke.block_len
        #
        if DEBUG_MODE:
            self._fock_basis = None#没有必要设置这个东西
            self._fock_dict = None#不必要设置

    @property
    def block_len(self):
        '''整体长度'''
        return self._block_len

    @property
    def leftblockextend(self):
        '''左侧的扩展后的block'''
        return self._lbke

    @property
    def rightblockextend(self):
        '''右侧的扩展后的block'''
        return self._rbke

    def __str__(self):
        template = 'SuperBlockExtend:\ndim: %d\n' % self._dim
        template += 'block_len: %d\n' % self._block_len
        randidx = numpy.random.randint(0, self._dim)
        for idx in self.iter_idx():
            if idx != 0 and idx != self._dim - 1 and idx != randidx:
                continue
            idxtup = self.idx_to_idxtuple(idx)
            template += '|%s,%s,%s,%s>\n' %\
                        (self._lblk.idx_to_state(idxtup[0]),\
                            self._lsite.idx_to_state(idxtup[1]),\
                                self._rsite.idx_to_state(idxtup[2]),\
                                    self._rblk.idx_to_state(idxtup[3]))
        return template

    def idx_to_idxtuple(self, idx):
        '''把一个编号转成四个数值的tuple'''
        scd = self.idx_to_sitecode(idx)
        idxtup = (scd[0], scd[1], scd[2], scd[3])
        return idxtup
