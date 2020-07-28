"""
在Left/Right-Block中的算符的基本属性
"""

import numpy
from basics.operator import Operator
from basics.basis import SiteBasis, ProdBasis
from .block import Block

class CreateOperator(Operator):
    """产生算符需要制定格子坐标，自旋，所在的基，
    stidx： 格子编号
    spin： 自旋+1或者-1
    bss： 所在的基
    val: 矩阵
    还有具体的矩阵，如果是在fock space上，可以直接调用create_from_sitebasis"""
    def __init__(self, stidx, spin, bss, val):
        '''这里只保存基础的属性'''
        super().__init__()
        self._stidx = stidx
        self._spin = spin
        self._basis = bss
        self._mat = val

    @staticmethod
    def create_from_sitebasis(stbss: SiteBasis, stidx, spin):
        '''从sitebasis创建升算符
        不能创建太大的site数量，没有使用SparseOperator
        ``````
        TODO: 是否要用稀疏的表示？
        '''
        if not stidx in stbss.sites:
            raise ValueError('create_from_sitebasis位置不在基里面')
        mat = numpy.zeros([stbss.dim, stbss.dim])
        stbsidx = stbss.sites.index(stidx)
        for rstc in stbss.iter_idx():
            #产生算符在sitebasis上，spin=1会把o->u, d->f，其余都是0
            #spin=-1会把o->d, u->f其余都是0
            stcode = stbss.idx_to_sitecode(rstc)
            if spin == 1:
                if stcode[stbsidx] == 0:
                    stcode[stbsidx] = 1
                elif stcode[stbsidx] == 2:
                    stcode[stbsidx] = 3
                else:#不是o或者d，直接回循环，不赋值
                    continue
            elif spin == -1:
                if stcode[stbsidx] == 0:
                    stcode[stbsidx] = 2
                elif stcode[stbsidx] == 1:
                    stcode[stbsidx] = 3
                else:#不是o或者u，直接回循环，不赋值
                    continue
            lstc = stbss.sitecode_to_idx(stcode)
            mat[lstc, rstc] = 1.
        return CreateOperator(stidx, spin, stbss, mat)

    @staticmethod
    def create_from_prodbasis(prodbss: ProdBasis, operdic, stidx, spin):
        '''从prodbasis创建产生算符'''
        mat = numpy.zeros([prodbss.dim, prodbss.dim])
        operlist = [operdic[pre].mat for pre in prodbss.prefixs]
        for ridx in prodbss.iter_idx():
            for lidx in prodbss.iter_idx():
                rstc = prodbss.idx_to_sitecode(ridx)
                lstc = prodbss.idx_to_sitecode(lidx)
                #21.15，把sitecode对应的乘起来
                mat[lidx, ridx] =\
                    numpy.prod([opm[lstc[opi], rstc[opi]]\
                        for opi, opm in enumerate(operlist)])
        return CreateOperator(stidx, spin, prodbss, mat)

    @staticmethod
    def create_from_merge_prod(
            blk: Block,
            prodop,
            phival,
            stidx, spin
        ):
        '''21.19式'''
        if not isinstance(phival, numpy.ndarray):
            phival = numpy.array(phival)
        pmat = prodop.mat
        mat1 = numpy.matmul(pmat, phival.transpose())
        newmat = numpy.matmul(phival, mat1)
        return CreateOperator(stidx, spin, blk, newmat)


    @property
    def mat(self):
        '''整个矩阵'''
        return self._mat

    def ele(self, lidx, ridx):
        '''返回某个矩阵元'''
        return self._mat[lidx, ridx]

    def __str__(self):
        template = "CreateOperator: \n"
        template += 'Basis: %s\n' % self._basis.__class__.__name__
        template += 'Site: %d Spin: %d\n' % (self._stidx, self._spin)
        template += 'mat:\n'
        template += str(self._mat)
        return template
