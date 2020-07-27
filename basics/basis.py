"""
粒子数表象下的基有关的功能
"""

from typing import List
import numpy

class Basis(object):
    """一个基，表示|prefix>...prefix=states[0],states[1],...,states[-1]
    """
    def __init__(
            self,
            prefix: str,#算符的名称
            states: List[str],#算符的分量，
        ):
        self._prefix = prefix
        self._states = states
        self._dim = len(states)
        # 一些扩展的内容
        self._states2idx = None

    def __str__(self):
        template = \
        'Basis:\n\
    |{name}> ... \n|{state1}>..|{stateN}> \n \
    dim={dim}'
        vals = {
            'name': self._prefix,
            'state1': self._states[0],
            'stateN': self._states[-1],
            'dim': self._dim
        }
        return template.format(**vals)

    @property
    def states(self):
        '''可能的态'''
        return self._states

    @property
    def dim(self):
        '''维度'''
        return self._dim

    @property
    def prefix(self):
        '''前缀'''
        return self._prefix

    def idx_to_state(self, idx):
        '''从编号到字符表示'''
        return self._states[idx]

    def state_to_idx(self, sta):
        '''从字符表示到编号'''
        if self._states2idx is None:
            self._states2idx = dict(
                [(sta, idx) for idx, sta in enumerate(self._states, 0)]
                )
        return self._states2idx[sta]

    def rdirect_product(self, bs2, newpre: str):
        '''将两个basis进行直积运算, bs2放在右边
        computational many particle physics (21.10)
        '''
        newstas = []
        for lowbit in self._states:
            for highbit in bs2.states:
                newstas.append('%s,%s' % (lowbit, highbit))
        newbss = Basis(newpre, newstas)
        return newbss

    def iter_idx(self):
        '''返回迭代器'''
        #for sta in self._states:
        #    yield self.state_to_idx(sta)
        for idx in range(self._dim):
            yield idx

class SiteBasis(Basis):
    """以格子的占据状态为基准的基
    这个SiteBasis主要的特色是其中的sites数组包含了这个基之中所包含的格子的编号
    每个格子都有相同的states，所有格子的占据状态形成的数组这里叫做sitecode，
    对应的字符串（所有格子的）这里叫做state，把最小的格子当作最低位，可以算出一个
    0～dim的整数，叫做idx。
    ``````
    这里最重要的功能就是idx_to_sitecode和sitecode_to_idx，
    还有实现右直积的rdirect_product
    """
    def __init__(self, prefix: str, sites, states: List):
        '''sites是格子的编号，可以是一个整数或者整数数组'''
        super().__init__(prefix, states)
        if isinstance(sites, int):
            self._sites = [sites]
        else:
            self._sites = sorted(sites)
        self._prefix = ','.join(['%s_%d' % (prefix, sidx) for sidx in self._sites])
        self._dim = numpy.power(len(states), len(self._sites))

    def __str__(self):
        template = \
        'Basis:\n\
    |{name}> ... |{state1}>..|{stateN}> \n \
    dim={dim}'
        vals = {
            'name': self._prefix,
            'state1': self.sitecode_to_state([0 for _ in self._sites]),
            'stateN': self.sitecode_to_state([-1 for _ in self._sites]),
            'dim': self._dim
        }
        return template.format(**vals)

    @property
    def sites(self):
        '''包含的格子的编号'''
        return self._sites

    def idx_to_sitecode(self, idx):
        '''从编号到每个格子的态，编号的方式是从小到大，小的是低位'''
        stbit = len(self._sites) - 1
        staidx = [0] * len(self._sites)
        num = idx
        while num > 0:
            bitden = numpy.power(len(self._states), stbit, dtype=numpy.float)
            bitnum = numpy.floor(num / bitden).astype(numpy.int)
            staidx[stbit] = bitnum
            num -= bitnum * bitden
            stbit -= 1
        return staidx

    def sitecode_to_idx(self, staidx):
        '''小的格子是低位'''
        result = 0
        rank = 1
        for val in staidx:
            result += rank * val
            rank *= len(self._states)
        return result

    def idx_to_state(self, idx):
        '''编号的方式是格子标号从小到大，小的是低位'''
        staidx = self.idx_to_sitecode(idx)
        return self.sitecode_to_state(staidx)

    def state_to_idx(self, sta: str):
        '''把一个字符转换成态'''
        #首先把state到数组中编号的计算清楚
        if self._states2idx is None:
            self._states2idx = dict(
                [(sta, idx) for idx, sta in enumerate(self._states, 0)]
                )
        sta_list = sta.split(',')
        stadic = {}
        for sta_idx in sta_list:
            sil = sta_idx.split('_')
            sta = sil[0]
            idx = int(sil[1])
            stadic[idx] = self._states2idx[sta]
        stcode = [0] * len(self._sites)
        for idx, stid in enumerate(self._sites, 0):
            stcode[idx] = stadic[stid]
        return self.sitecode_to_idx(stcode)

    def sitecode_to_state(self, staidx):
        '''staidx是每个格子的状态的编号，注意需要按照从小到大，需要是一个数组'''
        if len(staidx) != len(self._sites):
            raise ValueError('staidx和self._sites长度不一致')
        return ','.join(['%s_%d' % (self._states[sta], idx)\
            for sta, idx in zip(staidx, self._sites)])

    def rdirect_product(self, bs2, newpre):
        '''注意这时两个basis的states一定要是相等的'''
        if len(self._states) != len(self.states):
            raise NotImplementedError('暂时不实现不一样的格子')
        newbss = SiteBasis(newpre, self._sites + bs2.sites, self._states)
        return newbss

    def ldirect_product(self, bs2, newpre):
        '''注意这时两个basis的states一定要是相等的'''
        if len(self._states) != len(self.states):
            raise NotImplementedError('暂时不实现不一样的格子')
        newbss = SiteBasis(newpre, bs2.sites + self._sites, self._states)
        return newbss

    def iter_staidx(self):
        '''staidx的迭代器，以比较小的格子作为lowbit开始编码'''
        indicator = [0] * len(self._sites)
        indicator[0] = -1#循环前的准备
        stdim = len(self._states)
        #
        def __add_one():
            '''给指示器加1'''
            bitidx = 0
            while bitidx < len(self._sites):
                indicator[bitidx] += 1
                if indicator[bitidx] == stdim:#如果需要进位
                    indicator[bitidx] = 0
                    bitidx += 1#继续循环让更高位+1
                else:#如果不需要进位，直接反回成功
                    return True
            return False#如果一直在进位，就失败了

        while __add_one():
            yield indicator
