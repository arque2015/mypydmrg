'''和MPO算符有关的功能'''

import numpy
from . import WTensor, tensor_w_shrink_m
#from ..mps import MTensor
from ..mps.mps_state import MPSState

class MPOOperator():
    """一个MPO算符"""
    def __init__(
            self,
            leftinit: WTensor,
            rightinit: WTensor,
            tot_len
        ):
        self._tot_len = tot_len
        self._container = {}
        self._container[leftinit._idx] = leftinit
        self._container[rightinit._idx] = rightinit


    def __str__(self):
        template = '%s: \n' % self.__class__.__name__
        template += str(self._container)
        template += '\n'
        return template

    def add_site(self, tensor):
        '''添加一个site'''
        if tensor._idx in self._container:
            raise ValueError('这个格子已经存在了')
        self._container[tensor._idx] = tensor


    def apply_to_mps(self, mps: MPSState):
        '''作用到一个mps上，生成新的mps'''
        #TODO: 反对易的符号
        mps_dict = {}
        for idx in range(1, self._tot_len+1):
            nten = tensor_w_shrink_m(
                self._container[idx],
                mps._container[idx]
            )
            mps_dict[idx] = nten
        ret = MPSState(mps_dict[1], mps_dict[self._tot_len], self._tot_len)
        for idx in range(2, self._tot_len):
            ret.add_site(mps_dict[idx])
        return ret


def create_operator_of_site(stidx, length):
    '''某个格子上的产生算符'''
    mpo_dict = {}
    for idx in range(1, length+1):
        wmat = numpy.zeros([1, 4, 1, 4])
        if idx ==  stidx:
            #这个是o到u，delta_in在右边
            wmat[0, 1, 0, 0] = 1.
            #这个是d到f
            wmat[0, 3, 0, 2] = 1.
        else:
            for didx in range(4):
                wmat[0, didx, 0, didx] = 1.
        wten = WTensor(1, 1, idx, wmat)
        mpo_dict[idx] = wten
    #
    ret = MPOOperator(mpo_dict[1], mpo_dict[length], length)
    for idx in range(2, length):
        ret.add_site(mpo_dict[idx])
    return ret
