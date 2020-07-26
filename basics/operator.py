"""
和算符有关的功能
"""

import numpy
from .basis import Basis
from .states import States

class Operator(object):
    """基本的算符，不要直接调用这个Operator"""
    def __init__(self):
        self._basis = None
        self._ldim = 0
        self._rdim = 0
        self._val_iter = None

    def __str__(self):
        template = 'Operator %s\n' % self.__class__.__name__
        template += str(self._basis)
        if self._val_iter is None:
            raise NotImplementedError('%s没有实现val_iter' % self.__class__.__name__)
        eles = 0
        for idx, elepair in enumerate(self._val_iter, 0):
            template += '%d: %s\n' % (idx, str(self.ele(elepair[0], elepair[1])))
            eles = idx
            if eles >= 10:
                break
        if eles >= 10:
            template += '......\n'
        return template


    def ele(self, lidx, ridx):
        '''获得某个矩阵元'''
        raise NotImplementedError('Operator不实现这个功能')

    def _yield_val_iter(self):
        '''生成迭代数值的迭代器'''
        for lidx in range(self._ldim):
            for ridx in range(self._rdim):
                yield (lidx, ridx)


class DenseObservable(Operator):
    """直接用矩阵表示的比较稠密的算符"""
    def __init__(
            self,
            initbss: Basis,#最开始在那个基上
            initmat,#最开始的矩阵
        ):
        super().__init__()
        self._basis = initbss
        self._ldim = initbss.dim
        self._rdim = initbss.dim
        self._mat = numpy.zeros([self._ldim, self._rdim])
        self._mat += initmat
        self._val_iter = super()._yield_val_iter()
        #print(list(self._val_iter))


    def ele(self, lidx, ridx):
        return self._mat[lidx, ridx]

    @property
    def mat(self):
        '''矩阵表示'''
        return self._mat


class DenseTranslator(Operator):
    """直接用矩阵表示的稠密的转换基的算符"""
    def __init__(
            self,
            srcbss: Basis,#从srcbss转换到dstbss
            dstbss: Basis,#srcbss转换到dstbss
            initmat,#初始化的矩阵，每一行应该是新的基在旧的基上的分量
            initdic=None#可以不用初始化的矩阵，这里面每一个key应该是新的基的到旧的基的分量
            #优先使用initmat中的数值
        ):
        super().__init__()
        self._basis = srcbss
        self._ldim = dstbss.dim
        self._rdim = srcbss.dim
        self._val_iter = super()._yield_val_iter()
        #
        self._srcbasis = srcbss
        self._dstbasis = dstbss
        self._mat = numpy.zeros([dstbss.dim, srcbss.dim])
        if not initmat is None:
            self._mat += initmat
            if not initdic is None: 
                print('使用initmat中的设置，跳过initdic')

        if initmat is None:
            if initdic is None:
                raise ValueError('initmat initdic都是None')
            for key in range(self._dstbasis.dim):
                self._mat[key, :] = initdic[key]

    def ele(self, lidx, ridx):
        return self._mat[lidx, ridx]

    @property
    def mat(self):
        '''矩阵表示'''
        return self._mat

    def translate_state(self, sta: States):
        '''将sta转换到新的基上面'''
        vec = sta.components
        newvec = numpy.matmul(self._mat, vec)
        newsta = States(self._dstbasis, newvec)
        return newsta

    def translate_observable(self, obs: DenseObservable):
        '''将obs转换到新的基上面'''
        oldmat = obs.mat
        newmat = numpy.matmul(
            self._mat,
            numpy.matmul(oldmat, self._mat.transpose())
            )
        newobs = DenseObservable(self._dstbasis, newmat)
        return newobs

