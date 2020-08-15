"""包含MPO的一些功能"""

import numpy
from ..mps import MTensor

class WTensor():
    '''表示一个W，有四个指标\n
    in的状态，out的状态，还有两个是表示格子之间联系的\n
    标记的顺序是b_l-1, delta_l^out, b_l, delta_l^in\n
    '''
    def __init__(self, b_l_1, b_l, idx, entity):
        self._idx = idx
        if isinstance(entity, numpy.ndarray):
            self._entity = numpy.reshape(entity, [b_l_1, 4, b_l, 4])
        else:
            self._entity = numpy.array(entity)
            self._entity = numpy.reshape(self._entity, [b_l_1, 4, b_l, 4])
        self._tshape = numpy.shape(self._entity)


    def __str__(self):
        template = '%s: \n' % self.__class__.__name__
        template += str(self._tshape)
        template += '\n'
        return template


    def __repr__(self):
        return str(self)


def tensor_w_shrink_m(wten: WTensor, mten: MTensor):
    '''一个wtensor和一个mtensor合并'''
    #wtensor的形状是(b_l-1, delta_l^out, b_l, delta_l^in)
    #mtensor的形状是(a_l-1, delta_l^in, a_l)
    #最后形成的ntensor的形状是(b_l-1*a_l-1, delta_l^out, b_l*a_l)
    #先把wtensor整理成矩阵
    wshape = wten._tshape
    wmat = numpy.reshape(wten._entity,\
        [wshape[0]*wshape[1]*wshape[2], wshape[3]])
    #整理mmat
    mshape = mten.tensor_shape
    mmat = numpy.transpose(mten.tensor_entity, [1, 0, 2])
    mmat = numpy.reshape(mmat, [mshape[1], mshape[0]*mshape[2]])
    #收缩delta,nmat的形状是(b_l-1*delta_l^out*b_l, a_l-1*a_l)
    nmat = numpy.matmul(wmat, mmat)
    #整理nmat的形状，把乘在一起的分开
    nmat = numpy.reshape(nmat,\
        (wshape[0], wshape[1], wshape[2], mshape[0], mshape[2]))
    #调整顺序到需要的形状
    nmat = numpy.transpose(nmat, [0, 3, 1, 2, 4])
    nten = MTensor(
        wshape[0]*mshape[0], wshape[2]*mshape[2],
        mten.idx, nmat
    )
    return nten
