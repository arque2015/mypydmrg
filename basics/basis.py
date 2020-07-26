"""
粒子数表象下的基有关的功能
"""

from typing import List

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
    |{name}> ... |{state1}>..|{stateN}> \n \
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
        return len(self._states)

    def idx_to_state(self, idx):
        '''从编号到字符表示'''
        return self._states[idx]

    def state_to_idx(self, idx):
        '''从字符表示到编号'''
        if self._states2idx is None:
            self._states2idx = dict(
                [(sta, idx) for idx, sta in enumerate(self._states, 0)]
                )
        return self._states2idx[idx]

    def rdirect_product(self, bs2, newpre: str):
        '''将两个basis进行直积运算, bs2放在右边
        computational many particle physics (20.10)
        '''
        newstas = []
        for lowbit in self._states:
            for highbit in bs2.states:
                newstas.append('%s%s' % (lowbit, highbit))
        newbss = Basis(newpre, newstas)
        return newbss


class SiteBasis(Basis):
    """以格子的占据状态为基准的基"""
    def __init__(self, idx: int, states: List):
        '''idx是格子的编号'''
