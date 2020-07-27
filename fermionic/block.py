"""
left/right-block相关的功能
"""

import numpy
from basics.basis import SiteBasis, Basis, ProdBasis
from . import DEBUG_MODE

class Block(Basis):
    """可以用来表示LeftBlock或者RightBlock
    """
    def __init__(self, sitebss: SiteBasis, stanum, initmat=None):
        if not DEBUG_MODE and initmat is not None:
            print('不在debug_mode，initmat不会有作用')
        #
        self._fock_basis = sitebss
        self._dim = stanum
        self._block_len = len(sitebss.sites)
        self._postfix = ''
        _stridx = [str(idx) for idx in range(self._dim)]
        super().__init__('phi^%d' % len(sitebss.sites), _stridx)
        #
        if DEBUG_MODE and initmat is not None:
            self._fock_dict = dict(enumerate(initmat, 0))

    @property
    def block_len(self):
        '''这个LeftBlock有多少site在里面'''
        return self._block_len

    @property
    def fock_basis(self):
        '''这个block在哪个SiteBasis上'''
        return self._fock_basis

    @property
    def fock_dict(self):
        '''这个block在SiteBasis上的表示'''
        if not DEBUG_MODE:
            raise NotImplementedError('非debug模式不计算block')
        return self._fock_dict

    def __str__(self):
        template = '%s: |%s_%s>\n' %\
            (self.__class__.__name__, self._prefix, self._postfix)
        template += 'dim: %d\n' % self.dim
        for idx in self.iter_idx():
            if idx != 0 and idx != self._dim - 1:
                continue
            template += '|%s_%d>' % (self._prefix, idx)
            if not DEBUG_MODE:
                template += '\n'
                continue
            template += ' :['
            sitebss_arr = self._fock_dict[idx]
            for idx2 in range(self._fock_basis.dim):
                if (idx2 % 4) == 0:
                    template += '\n'
                template += '%.4f |%s>\t' %\
                    (sitebss_arr[idx2], self._fock_basis.idx_to_state(idx2))
            template += '\n]\n'
        return template

class LeftBlock(Block):
    """表示LeftBlock computational many particle physics (21.6)
    这个对应的是等式左边，有个下角标alpha
    依旧是每一行是一个新的基，每一行的内容是它在sitebasis上面的分量，
    所以列必须是sitebasis.dim
    """
    def __init__(self, sitebss: SiteBasis, stanum, initmat=None):
        super().__init__(sitebss, stanum, initmat)
        self._postfix = 'alpha'

    def rdirect_product(self, bs2: SiteBasis, newpre=''):#pylint: disable=unused-argument
        '''将现在的block扩大一个site，（21.10）式
        注意|phi> X |s_n> = |phi,s_n> = sum_i a_i|s..s_i, s_n>
        而对于sitebasis是从小到大判断低位和高位的
        '''
        return LeftBlockExtend(self, bs2)


class RightBlock(Block):
    """表示RightBlock computational many particle physics (21.7)
    有个下角标beta，和LeftBlock基本上是一样的
    依旧是每一行是一个新的基，每一行的内容是它在sitebasis上面的分量，
    所以列必须是sitebasis.dim
    """
    def __init__(self, sitebss: SiteBasis, stanum, initmat=None):
        super().__init__(sitebss, stanum, initmat)
        self._postfix = 'beta'


    def ldirect_product(self, bs2: SiteBasis, newpre=''):#pylint: disable=unused-argument
        '''将现在的block扩大一个site，（21.10）式
        注意|phi> X |s_n> = |phi,s_n> = sum_i a_i|s..s_i, s_n>
        而对于sitebasis是从小到大判断低位和高位的
        '''
        return RightBlockExtend(self, bs2)


class LeftBlockExtend(ProdBasis):
    """扩大了一个site以后的LeftBlock，这个是（21.10）的左手边
    这时idxpair是类似SiteBasis中sitecode的东西"""
    def __init__(
            self,
            lblk: LeftBlock,
            stbss: SiteBasis
        ):
        # 基本的属性
        self._lblk = lblk
        self._stbss = stbss
        #
        stadic = {}
        stadic[lblk.prefix] = lblk.states
        stadic[stbss.prefix] = stbss.states
        super().__init__(stadic)
        # 一些额外的属性
        self._block_len = lblk.block_len + len(stbss.sites)
        self._fock_basis = lblk.fock_basis.rdirect_product(stbss, 's')
        if DEBUG_MODE:
            self._fock_dict = numpy.zeros([self.dim, self._fock_basis.dim])
            # |idx1, idx2>这个依旧以左边为低位
            # 而|s..s_n, s_n+1>这个可以通过oldidx + s_n+1 * (st.dim)^n，直接计算出新的
            # fock_basis中的newidx
            #for idx2 in stbss.iter_idx():
            #    for idx1 in lblk.iter_idx():
            for idx in range(self._dim):
                idx1, idx2 = self.idx_to_idxpair(idx)
                comps = lblk.fock_dict[idx1]
                oldlen = lblk.fock_basis.dim
                for oldidx in range(oldlen):
                    #这个oldidx就是按照格子编号小为低位，编号大为高位计算的
                    #直接加上新加的格子为最高位就可以了
                    newidx = oldidx + idx2 * oldlen
                    self._fock_dict[idx][newidx] = comps[oldidx]

    @property
    def lblk(self):
        '''LeftBlock'''
        return self._lblk

    @property
    def stbss(self):
        '''新加的site'''
        return self._stbss

    @property
    def block_len(self):
        '''block中含有的site数量'''
        return self._block_len

    def idx_to_idxpair(self, idx):
        '''将idx转换成idxpair'''
        idxf = numpy.float(idx)
        idx2 = numpy.floor(idxf / self._lblk.dim).astype(numpy.int)
        idx1 = idx - idx2 * self._lblk.dim
        return idx1, idx2

    def idxpair_to_idx(self, idx1, idx2):
        '''将idxpair转换成idx'''
        return idx1 + idx2 * self._lblk.dim

    def __str__(self):
        template = 'LeftBlockExtend: \ndim: %d\n' % self._dim
        randshow = numpy.random.randint(0, self._dim)
        for idx in self.iter_idx():
            if idx != 0 and idx != self._dim - 1 and idx != randshow:
                continue
            template += '%d %s' % (idx, self.idx_to_state(idx))
            if not DEBUG_MODE:
                template += '\n'
                continue
            for idx2 in range(self._fock_basis.dim):
                if (idx2 % 4) == 0:
                    template += '\n'
                template += '%.4f |%s>\t' %\
                    (self._fock_dict[idx][idx2], self._fock_basis.idx_to_state(idx2))
            template += '\n'
        return template




class RightBlockExtend(ProdBasis):
    """扩大了一个site以后的RightBlock"""
    def __init__(
            self,
            rblk: RightBlock,
            stbss: SiteBasis
        ):
        # 基本的属性
        self._rblk = rblk
        self._stbss = stbss
        # 和LeftBlockExtend不同的是，这时新加的site是低位
        # 依旧是左边是低位
        stadic = {}
        stadic[stbss.prefix] = stbss.states
        stadic[rblk.prefix] = rblk.states
        #
        super().__init__(stadic)
        # 一些额外的属性
        self._block_len = rblk.block_len + len(stbss.sites)
        self._fock_basis = rblk.fock_basis.ldirect_product(stbss, 's')
        # 
        if DEBUG_MODE:
            self._fock_dict = numpy.zeros([self.dim, self._fock_basis.dim])
            # |idx1, idx2>这个依旧以左边为低位
            # 而|s..s_n, s_n+1>这个可以通过s_n+1 + oldidx * st.dim，直接计算出新的
            # fock_basis中的newidx
            for idx in range(self._dim):
                idx1, idx2 = self.idx_to_idxpair(idx)
                comps = rblk.fock_dict[idx2]
                oldlen = rblk.fock_basis.dim
                for oldidx in range(oldlen):
                    #这个oldidx是右边的block的sitebasis的维度
                    #在新的sitebasis中，oldidx是高位
                    newidx = idx1 + oldidx * stbss.dim
                    self._fock_dict[idx][newidx] = comps[oldidx]

    @property
    def rblk(self):
        '''RightBlock'''
        return self._rblk

    @property
    def stbss(self):
        '''增加的site'''
        return self._stbss

    @property
    def block_len(self):
        '''block中含有的site数量'''
        return self._block_len

    def idx_to_idxpair(self, idx):
        '''将idx转换成idxpair'''
        idxf = numpy.float(idx)
        idx2 = numpy.floor(idxf / self._stbss.dim).astype(numpy.int)
        idx1 = idx - idx2 * self._stbss.dim
        return idx1, idx2

    def idxpair_to_idx(self, idx1, idx2):
        '''将idxpair转换成idx'''
        return idx1 + idx2 * self._stbss.dim

    def __str__(self):
        template = 'RightBlockExtend: \ndim: %d\n' % self._dim
        randshow = numpy.random.randint(0, self._dim)
        for idx in self.iter_idx():
            if idx != 0 and idx != self._dim - 1 and idx != randshow:
                continue
            template += '%d %s' % (idx, self.idx_to_state(idx))
            for idx2 in range(self._fock_basis.dim):
                if (idx2 % 4) == 0:
                    template += '\n'
                template += '%.4f |%s>\t' %\
                    (self._fock_dict[idx][idx2], self._fock_basis.idx_to_state(idx2))
            template += '\n'
        return template
