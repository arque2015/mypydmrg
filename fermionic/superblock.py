"""
superblock有关的功能
computational many particle physics （21.23） （21.8）
"""

import numpy
from basics.basis import Basis
from .block import LeftBlockExtend, RightBlockExtend

class SuperBlockExtend(Basis):
    """将一个LeftBlockExtend和一个RightBlockExtend
    直积成一个superblock"""
    def __init__(self, lbke: LeftBlockExtend, rbke: RightBlockExtend):
        self._lbke = lbke
        self._rbke = rbke
        #
        self._stapairs = []
        llen = lbke.lblk.block_len
        rlen = rbke.rblk.block_len
        for idx in range(lbke.dim * rbke.dim):
            idx1, idx2 = self.idx_to_idxpair(idx)
            lblkidx, lstidx = self._lbke.idx_to_idxpair(idx1)
            rstidx, rblkidx = self._rbke.idx_to_idxpair(idx2)
            self._stapairs.append(
                'phi^%d_%d,%s,%s,phi^%d_%d' %\
                    (llen, lblkidx, lbke.stbss.idx_to_state(lstidx),\
                        rbke.stbss.idx_to_state(rstidx), rlen, rblkidx)
            )
        #
        super().__init__(
            'phi^%d_alpha,%s,%s,phi^%d_beta' %\
                (llen, lbke.stbss.prefix, rbke.stbss.prefix, rlen),
            self._stapairs
            )
        #
        self._block_len = lbke.block_len + rbke.block_len
        self._fock_basis = None#没有必要设置这个东西
        self._fock_dict = None#不必要设置

    @property
    def block_len(self):
        '''整体长度'''
        return self._block_len

    def __str__(self):
        template = 'SuperBlockExtend:\ndim: %d' % self._dim
        #randshow = numpy.random.randint(0, self._dim)
        llen = self._lbke.lblk.block_len
        rlen = self._rbke.rblk.block_len
        for idx in self.iter_idx():
            idx1, idx2 = self.idx_to_idxpair(idx)
            lblkidx, lstidx = self._lbke.idx_to_idxpair(idx1)
            rstidx, rblkidx = self._rbke.idx_to_idxpair(idx2)
            template += 'phi^%d_%d,%s,%s,phi^%d_%d\n' %\
                        (llen, lblkidx, self._lbke.stbss.idx_to_state(lstidx),\
                            self._rbke.stbss.idx_to_state(rstidx), rlen, rblkidx)
        return template

    def idx_to_idxpair(self, idx):
        '''Left是低位，Right是高位'''
        idxf = numpy.float(idx)
        idx2 = numpy.floor(idxf / self._lbke.dim).astype(numpy.int)
        idx1 = idx - idx2 * self._lbke.dim
        return idx1, idx2

    def idxpair_to_idx(self, idx1, idx2):
        '''这时的idxpair里面是LeftBlockExtend和RightBlockExtend的idx'''
        return idx1 + idx2 * self._lbke.dim
