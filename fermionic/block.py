"""
left/right-block相关的功能
"""

import numpy
from basics.basis import SiteBasis, Basis

class Block(Basis):
    """一个block是包含了多个site的basis
    创建一个block需要一个指明block中的分量由哪些SiteBasis中的基构成
    比如[[0.5, 0, 0.5, 0], [...]]这时，二维数组的第一个指标是block的指标
    二维数组的第二个指标会按照SiteBasis中的idx判断是哪个基
    注意多使用Basis中的idx_to_state和state_to_idx而不要自己管理顺序
    也不要和SiteBasis中的sitecode产生混淆，block中并没有和site有关系的
    编号，只有一个总共的长度
    """
    def __init__(self, prefix, sitebss: SiteBasis, initmat):
        ''''''
        blocksta = ['%s_%d' % (prefix, idx1) for idx1 in range(len(initmat))]
        super().__init__(prefix, blocksta)
        self._fock_basis = sitebss
        self._fock_dict = dict(enumerate(initmat, 0))
        self._block_len = len(sitebss.sites)

    @property
    def block_len(self):
        '''block中有几个site'''
        return self._block_len

    @property
    def fock_basis(self):
        '''block基于的sitebasis'''
        return self._fock_basis

    @property
    def fock_dict(self):
        '''block在sitebasis上具体的表示'''
        return self._fock_dict

class LeftBlock(Block):
    """表示LeftBlock computational many particle physics (21.6)
    有个下角标alpha
    依旧是每一行是一个新的基，每一行的内容是它在sitebasis上面的分量，
    所以列必须是sitebasis.dim
    """
    def __init__(self, sitebss: SiteBasis, initmat):
        super().__init__('phi^%d' % len(sitebss.sites), sitebss, initmat)

    def idx_to_sitebasis(self, idx):
        '''返回某个基在sitebasis下的表示'''
        return self._fock_dict[idx]

    def __str__(self):
        template = 'LeftBlock: |%s_alpha>\n' % self._prefix
        template += 'dim: %d\n' % self.dim
        for idx in self.iter_idx():
            if idx != 0 and idx != self._dim - 1:
                continue
            template += '|%s_%d> :' % (self._prefix, idx)
            template += '['
            sitebss_arr = self.idx_to_sitebasis(idx)
            for idx2 in range(self._fock_basis.dim):
                if (idx2 % 4) == 0:
                    template += '\n'
                template += '%.4f |%s>\t' %\
                    (sitebss_arr[idx2], self._fock_basis.idx_to_state(idx2))
            template += '\n]\n'
        return template

    def rdirect_product(self, bs2: SiteBasis, newpre=''):#pylint: disable=unused-argument
        '''将现在的block扩大一个site，（21.10）式
        注意|phi> X |s_n> = |phi,s_n> = sum_i a_i|s..s_i, s_n>
        而对于sitebasis是从小到大判断低位和高位的
        '''
        return LeftBlockExtend(self._prefix, bs2.prefix, self, bs2)


class LeftBlockExtend(Basis):
    """扩大了一个site以后的LeftBlock，这个是（21.10）的左手边
    这时idxpair是类似SiteBasis中sitecode的东西"""
    def __init__(
            self,
            prefix1, prefix2,
            lblk: LeftBlock,
            stbss: SiteBasis
        ):
        # 基本的属性
        self._lblk = lblk
        self._stbss = stbss
        #self._idxpairs = []
        self._stapairs = []
        for idx in range(lblk.dim * stbss.dim):
            idx1, idx2 = self.idx_to_idxpair(idx)
            self._stapairs.append(
                '%s_%d,%s' % (prefix1, idx1, stbss.idx_to_state(idx2))
            )
        #for idx2 in stbss.iter_idx():
        #    for idx1 in lblk.iter_idx():
        #        self._idxpairs.append((idx1, idx2))
        #        self._stapairs.append(
        #            '%s_%d,%s' % (prefix1, idx1, stbss.idx_to_state(idx2))
        #        )
        super().__init__(prefix1+'_alpha,'+prefix2, self._stapairs)
        # 一些额外的属性
        self._block_len = lblk.block_len + len(stbss.sites)
        self._fock_basis = lblk.fock_basis.rdirect_product(stbss, 's')
        # fock_dict
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
            for idx2 in range(self._fock_basis.dim):
                if (idx2 % 4) == 0:
                    template += '\n'
                template += '%.4f |%s>\t' %\
                    (self._fock_dict[idx][idx2], self._fock_basis.idx_to_state(idx2))
            template += '\n'
        return template


class RightBlock(Block):
    """表示RightBlock computational many particle physics (21.7)
    有个下角标beta，和LeftBlock基本上是一样的
    依旧是每一行是一个新的基，每一行的内容是它在sitebasis上面的分量，
    所以列必须是sitebasis.dim
    """
    def __init__(self, sitebss: SiteBasis, initmat):
        super().__init__('phi^%d' % len(sitebss.sites), sitebss, initmat)

    def idx_to_sitebasis(self, idx):
        '''返回某个基在sitebasis下的表示'''
        return self._fock_dict[idx]

    def __str__(self):
        template = 'RightBlock: |%s_beta>\n' % self._prefix
        template += 'dim: %d\n' % self.dim
        for idx in self.iter_idx():
            if idx != 0 and idx != self._dim - 1:
                continue
            template += '|%s_%d> :' % (self._prefix, idx)
            template += '['
            sitebss_arr = self.idx_to_sitebasis(idx)
            for idx2 in range(self._fock_basis.dim):
                if (idx2 % 4) == 0:
                    template += '\n'
                template += '%.4f |%s>\t' %\
                    (sitebss_arr[idx2], self._fock_basis.idx_to_state(idx2))
            template += '\n]\n'
        return template

    def ldirect_product(self, bs2: SiteBasis, newpre=''):#pylint: disable=unused-argument
        '''将现在的block扩大一个site，（21.10）式
        注意|phi> X |s_n> = |phi,s_n> = sum_i a_i|s..s_i, s_n>
        而对于sitebasis是从小到大判断低位和高位的
        '''
        return RightBlockExtend(self._prefix, bs2.prefix, self, bs2)

class RightBlockExtend(Basis):
    """扩大了一个site以后的RightBlock"""
    def __init__(
            self,
            prefix1, prefix2,
            rblk: RightBlock,
            stbss: SiteBasis
        ):
        # 基本的属性
        self._rblk = rblk
        self._stbss = stbss
        # 和LeftBlockExtend不同的是，这时新加的site是低位
        # 依旧是左边是低位
        self._stapairs = []
        for idx in range(rblk.dim * stbss.dim):
            idx1, idx2 = self.idx_to_idxpair(idx)
            self._stapairs.append(
                '%s,%s_%d' % (stbss.idx_to_state(idx1), prefix1, idx2)
            )
        #
        super().__init__(prefix2+','+prefix1+'_beta', self._stapairs)
        # 一些额外的属性
        self._block_len = rblk.block_len + len(stbss.sites)
        self._fock_basis = rblk.fock_basis.ldirect_product(stbss, 's')
        # fock_dict
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
