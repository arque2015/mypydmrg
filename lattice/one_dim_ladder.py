"""一维的台阶"""

import numpy
from . import BaseModel


class HubbardLadder(BaseModel):
    r"""一维的Ladder
    3 - - 6 - -
    |\    |\
    | 2   | 5
    |/    |/
    1 - - 4 - -
    ``````
    一共需要3个t，pbc是需要满足的
    """
    def __init__(self, lenx, coef_u, alpha, coef_mu=0.0):
        super().__init__(list(range(1, 3 * lenx + 1)))
        self._lenx = lenx
        self._coef_u = coef_u
        self._coef_mu = {}
        self._alpha = alpha
        #先设置竖折的几个bond
        for idx_leg1 in range(lenx):
            #这个是下面一层的格子
            stidx = 3 * idx_leg1 + 1
            self._bonds[stidx] = [stidx + 1, stidx + 2]
            self._coef_mu[stidx] = coef_mu
            #上面一层的格子
            self._bonds[stidx + 2] = [stidx, stidx + 1]
            self._coef_mu[stidx + 2] = coef_mu
            #中间一层的格子
            self._bonds[stidx + 1] = [stidx, stidx + 2]
            self._coef_mu[stidx + 1] = coef_mu
        #再设置横着的格子
        for idx_leg1 in range(1, lenx-1):
            stidx = 3 * idx_leg1 + 1
            self._bonds[stidx].extend([stidx + 3, stidx - 3])
            self._bonds[stidx + 2].extend([stidx + 5, stidx - 1])
        #再加上两边的
        if lenx == 2:
            self._bonds[1].append(4)
            self._bonds[4].append(1)
            self._bonds[3].append(6)
            self._bonds[6].append(3)
        elif lenx > 2:
            self._bonds[1].extend([4, lenx*3-2])
            self._bonds[lenx*3-2].extend([1, lenx*3-5])
            self._bonds[3].extend([6, lenx*3])
            self._bonds[lenx*3].extend([3, lenx*3-3])


    def get_t_coef(self, st1, st2):
        '''得到bond的系数'''
        if st2 not in self.get_site_bonds(st1):
            raise ValueError('st1和st2之间没有bond')
        kind1 = st1 % 3
        kind2 = st2 % 3
        kindpair = sorted([kind1, kind2])
        #最下面那一列是1，中间是2，上面是3
        #如果两个相同就是横着的
        if kindpair[0] == kindpair[1]:
            return 1.0
        #如果有一个是2就是斜着的
        if kindpair[0] == 2 or kindpair[1] == 2:
            return numpy.sqrt(3. + self._alpha)
        #以上都不是只有竖着的
        return numpy.sqrt(3.)
