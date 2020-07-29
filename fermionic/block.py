"""
left/right-block相关的功能
"""

import numpy
from basics.basis import SiteBasis, Basis, ProdBasis
from . import DEBUG_MODE

class Block(Basis):
    """可以用来表示LeftBlock或者RightBlock
    """
    def __init__(self, sitebss: SiteBasis, stanum, initmat, partnum):
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
        #computional many particle physics (21.6) (21,7)
        #需要有一个phi(alpha, n)来维持和上一个block的关系
        #从而实现（21.19）的算符的传递
        self._sub_block = None
        self._sub_phival = None
        #在(21.6)(21.7)中这个phival会有一些额外的限制，这里把
        #Sz量子数考虑上，因为这涉及费米符号问题
        self._spin_nums = None
        if partnum is not None:
            if not isinstance(partnum, numpy.ndarray):
                self._spin_nums = numpy.array(partnum)
            else:
                self._spin_nums = partnum

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

    @property
    def spin_nums(self):
        '''自旋本征值，Block应该取和自旋量子数算符的共同本正态'''
        return self._spin_nums

    def __str__(self):
        template = '%s: |%s_%s>\n' %\
            (self.__class__.__name__, self._prefix, self._postfix)
        template += 'dim: %d\n' % self.dim
        template += 'block_len: %d\n' % self._block_len
        for idx in self.iter_idx():
            if idx != 0 and idx != self._dim - 1:
                continue
            template += '|%s_%d>\n' % (self._prefix, idx)
            if self._spin_nums is not None:
                template += 'particle number: %s\n' % self._spin_nums[idx]
            if not DEBUG_MODE:
                template += '\n'
                continue
            template += '['
            sitebss_arr = self._fock_dict[idx]
            for idx2 in range(self._fock_basis.dim):
                if (idx2 % 4) == 0:
                    template += '\n'
                template += '%.4f |%s>\t' %\
                    (sitebss_arr[idx2], self._fock_basis.idx_to_state(idx2))
            template += '\n]\n'
        return template

    def set_sub_block(self, blkext, phival):
        '''设置sub_block
        phival是一个二维数组，第一个指标是和自身的维度一样的，
        代表的是21.6，中左侧的alpha，第二个指标是（alpha', s^n)，
        大小和LeftBlockExtend.dim一样，
        具体数值是LeftBlockExtend中的idx_to_idxpair拆分出来的。
        RightBlock和LeftBlock是类似的，但是block和site的顺序是不一样的
        '''
        self._sub_block = blkext
        if DEBUG_MODE:
            self._sub_phival = phival


class LeftBlock(Block):
    """表示LeftBlock computational many particle physics (21.6)
    这个对应的是等式左边，有个下角标alpha
    依旧是每一行是一个新的基，每一行的内容是它在sitebasis上面的分量，
    所以列必须是sitebasis.dim
    """
    def __init__(self, sitebss: SiteBasis, stanum, initmat=None, partnum=None):
        super().__init__(sitebss, stanum, initmat, partnum)
        self._postfix = 'alpha'

    def rdirect_product(self, bs2: SiteBasis, newpre=''):#pylint: disable=unused-argument
        '''将现在的block扩大一个site，（21.10）式
        注意|phi> X |s_n> = |phi,s_n> = sum_i a_i|s..s_i, s_n>
        而对于sitebasis是从小到大判断低位和高位的
        ``````
        注意在SiteBasis中会进行格子编号的排序，这里一定要右作用一个更大的
        注意使用单个site的SiteBasis
        '''
        return LeftBlockExtend(self, bs2)


class RightBlock(Block):
    """表示RightBlock computational many particle physics (21.7)
    有个下角标beta，和LeftBlock基本上是一样的
    依旧是每一行是一个新的基，每一行的内容是它在sitebasis上面的分量，
    所以列必须是sitebasis.dim
    """
    def __init__(self, sitebss: SiteBasis, stanum, initmat=None, partnum=None):
        super().__init__(sitebss, stanum, initmat, partnum)
        self._postfix = 'beta'


    def ldirect_product(self, bs2: SiteBasis, newpre=''):#pylint: disable=unused-argument
        '''将现在的block扩大一个site，（21.10）式
        注意|phi> X |s_n> = |phi,s_n> = sum_i a_i|s..s_i, s_n>
        而对于sitebasis是从小到大判断低位和高位的
        ``````
        注意在SiteBasis中会进行格子编号的排序，这里一定要左作用一个更小的
        注意使用单个site的SiteBasis
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
        self._spin_nums = None#spin_nums被调用时初始化，见spin_nums
        #self._site_nums = site_nums
        #
        stadic = {}
        stadic[lblk.prefix] = lblk.states
        stadic[stbss.prefix] = stbss.states#SiteBasis的states是单个site的所有状态
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

    @property
    def spin_nums(self):
        '''对应idx，每个态的自旋数'''
        site_nums = self._stbss.partinum
        if self._spin_nums is not None:
            return self._spin_nums
        #如果没有self._spin_nums
        #初始化self._spin_nums
        if self._lblk.spin_nums is None:
            raise ValueError('LeftBlockExtend: lblk的spin_num没有设置')
        if not isinstance(self._lblk.spin_nums, numpy.ndarray):
            raise ValueError('LeftBlockExtend: lblk的spin_num不是ndarray')
        _spin_nums = numpy.zeros([self.dim, 2])
        if not isinstance(site_nums, numpy.ndarray):
            site_nums = numpy.array(site_nums)
        for idx in self.iter_idx():
            idx1, idx2 = self.idx_to_idxpair(idx)
            _spin_nums[idx] = self._lblk.spin_nums[idx1] + site_nums[idx2]
        self._spin_nums = _spin_nums
        return _spin_nums


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
        template += 'block_len: %d\n' % self._block_len
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


    def merge_to_block(self, phival, pnum_list=None):
        '''将这个|phi^n-1>X|s^n>合并成|phi^n>
        phival需要是一个二维数组，行数是合并后保留的基的数量，
        列数需要与self.dim一致，注意在计算phival的过程中，
        需要用到phi^n-1, s^n的时候，尽量用self.idx_to_idxpair
        保证结果的一致性
        '''
        newstbs = self._fock_basis
        stnum = len(phival)
        newmat = None
        if DEBUG_MODE:
            #新的initmat应该包含stnum行，fock_basis.dim列
            #new_fock_dict[phi^n, sitebasis] =\
            #sum_phi^n-1{phival[phi^n, phi^n-1] * fock_dict[phi^n-1, sitebasis]}
            newmat = numpy.matmul(phival, self._fock_dict)
        newlb = LeftBlock(newstbs, stnum, initmat=newmat, partnum=pnum_list)
        newlb.set_sub_block(self, phival)
        return newlb


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
        stadic[stbss.prefix] = stbss.states#SiteBasis的states是单个site的所有状态
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
        template += 'block_len: %d\n' % self._block_len
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

    def merge_to_block(self, phival, pnum_list=None):
        '''将这个|s^n-1>X|phi^n>合并成|phi^n-1>
        phival需要是一个二维数组，行数是合并后保留的基的数量，
        列数需要与self.dim一致，注意在计算phival的过程中，
        需要用到phi^n, s^n-1的时候，尽量用self.idx_to_idxpair
        保证结果的一致性
        '''
        newstbs = self._fock_basis
        stnum = len(phival)
        newmat = None
        if DEBUG_MODE:
            #新的initmat应该包含stnum行，fock_basis.dim列
            #new_fock_dict[phi^n-1, sitebasis] =\
            #sum_phi^n{phival[phi^n-1, phi^n] * fock_dict[phi^n, sitebasis]}
            newmat = numpy.matmul(phival, self._fock_dict)
        newrb = RightBlock(newstbs, stnum, initmat=newmat, partnum=pnum_list)
        newrb.set_sub_block(self, phival)
        return newrb
