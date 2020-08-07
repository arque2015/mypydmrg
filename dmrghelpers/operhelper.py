"""
简化算符的创建与更新
"""

from collections import namedtuple
import numpy
from basics.basis import SiteBasis
from fermionic.block import LeftBlockExtend, LeftBlock
from fermionic.block import RightBlockExtend, RightBlock
from fermionic.baseop import BaseOperator
#from .sitehelper import STATE_PARTINUM


SINGLE_SITE_CREATE_SPINUP = numpy.array(
    [
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 1., 0.]
    ]
)

SINGLE_SITE_CREATE_SPINDOWN = numpy.array(
    [
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., -1., 0., 0.]
    ]
)

SINGLE_SITE_INTERACT_U = numpy.array(
    [
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 1.]
    ]
)

#SINGLE_SITE_NUMBER_SPINUP = numpy.array(
#    [
#        [0., 0., 0., 0.],
#        [0., 1., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 1.]
#    ]
#)

OPERTUP = namedtuple('opertup', ['isferm', 'spin', 'mat'])
class OperFactory(object):
    '''算符的种类'''
    @staticmethod
    def create_spinup():
        '''自旋向上的产生算符'''
        return OPERTUP(
            isferm=True,
            spin=1,
            mat=SINGLE_SITE_CREATE_SPINUP
        )

    @staticmethod
    def create_spindown():
        '''自旋向下的产生算符'''
        return OPERTUP(
            isferm=True,
            spin=-1,
            mat=SINGLE_SITE_CREATE_SPINDOWN
        )

    @staticmethod
    def create_u():
        '''产生U算符'''
        return OPERTUP(
            isferm=False,
            spin=0,
            mat=SINGLE_SITE_INTERACT_U
        )

    #@staticmethod
    #def number_spinup():
    #    '''自旋向上的粒子数量'''
    #    return OPERTUP(
    #        isferm=False,
    #        spinsector=1,
    #        mat=SINGLE_SITE_NUMBER_SPINUP
    #    )

def create_operator_of_site(basis: SiteBasis, tup: OPERTUP):
    '''从一个site创建，spin = 1代表向上，spin=-1代表向下
    '''
    if len(basis.sites) > 1:
        raise NotImplementedError('多个格子的不去实现')
    stidx = basis.sites[0]
    return BaseOperator(stidx, basis, tup.isferm, tup.mat, spin=tup.spin)


def leftblock_extend_oper(
        leftext: LeftBlockExtend,
        oper: BaseOperator
    ):
    '''oper需要是在leftext.lbkl上面的算符
    这个函数把oper扩展到leftext上面
    这里利用的是21.15左右的公式
    '''
    #
    opdim = oper.basis.dim
    if leftext.lblk.dim != opdim:
        raise ValueError('LeftBlockExtend.lblk.dim和oper的dim对应不上')
    #在leftblock中的算符是已经把符号处理好的
    #直接扩展到新的基上就行了
    mat = numpy.zeros([leftext.dim, leftext.dim])
    #只有idx2相等的时候才能有数值
    mat[0:opdim, 0:opdim] = oper.mat
    mat[opdim:2*opdim, opdim:2*opdim] = oper.mat
    mat[2*opdim:3*opdim, 2*opdim:3*opdim] = oper.mat
    mat[3*opdim:, 3*opdim:] = oper.mat
    return BaseOperator(oper.siteidx, leftext, oper.isferm, mat, spin=oper.spin)


def leftsite_extend_oper(
        leftext: LeftBlockExtend,
        oper: BaseOperator
    ):
    '''将LeftBlockExtend.stbss上面的算符扩展到LeftBlockExtend上
    这里利用的是21.15左右的公式，加上了反对易
    '''
    opdim = oper.basis.dim
    if leftext.stbss.dim != opdim:
        raise ValueError('oper.basis.dim和LeftBlockExtend.stbss.dim对应不上')
    #需要处理反对易的符号
    #|A_1,A_2,..A_n> = C_1C_2..C_n|0>
    #C_n|A_1,A_2..> = C_nC_1C_2..|0> = (-1)^a C_1C_2..C_n|0>
    #其中a的数量是phi^n-1中的粒子数，因为phi^n-1是和粒子数的共同本征态
    mat = numpy.zeros([leftext.dim, leftext.dim])
    for ridx in leftext.iter_idx():
        for lidx in leftext.iter_idx():
            rbkid, rstid = leftext.idx_to_idxpair(ridx)
            lbkid, lstid = leftext.idx_to_idxpair(lidx)
            #这个时候leftblock是单位算符，只有lbkid == rbkid时才有数值
            if lbkid != rbkid:
                continue
            #计算block中一共有多少粒子
            _partnum = numpy.sum(leftext.lblk.spin_nums[rbkid])
            if oper.isferm:
                _sign = 1. if _partnum % 2 == 0 else -1.
                mat[lidx, ridx] = _sign * oper.mat[lstid, rstid]
            else:
                mat[lidx, ridx] = oper.mat[lstid, rstid]
    return BaseOperator(oper.siteidx, leftext, oper.isferm, mat, spin=oper.spin)


def update_leftblockextend_oper(
        newlbk: LeftBlock,
        oper: BaseOperator,
        phival
    ):
    '''利用21.19左右的式子'''
    if not isinstance(phival, numpy.ndarray):
        raise ValueError('phival不是ndarray')
    mat = numpy.matmul(oper.mat, phival.transpose())
    mat = numpy.matmul(phival, mat)
    return BaseOperator(oper.siteidx, newlbk, oper.isferm, mat, spin=oper.spin)


def rightblock_extend_oper(
        rightext: RightBlockExtend,
        oper: BaseOperator
    ):
    '''把rightblock.rblk下的算符扩展到
    rightblockextend上面
    '''
    opdim = oper.basis.dim
    if rightext.rblk.dim != opdim:
        raise ValueError('oper.basis.dim和RightBlockExtend.rblk.dim对应不上')
    #需要处理反对易的符号
    #|A_1,A_2,..A_n> = C_1C_2..C_n|0>
    #在Cm|phi^n_beta> = |phi^n_beta'>的情况下
    #Cm|s^n-1,phi^n_beta> = -C^n-1Cm|0, phi^n_beta>
    #所以在扩展rightblock中的算符的时候，要看n-1上面有几个粒子
    mat = numpy.zeros([rightext.dim, rightext.dim])
    for ridx in rightext.iter_idx():
        for lidx in rightext.iter_idx():
            rstid, rbkid = rightext.idx_to_idxpair(ridx)
            lstid, lbkid = rightext.idx_to_idxpair(lidx)
            #这个时候rightsite是单位算符，只有lstid == rstid时才有数值
            if lstid != rstid:
                continue
            #计算site中一共有多少粒子
            _partnum = numpy.sum(rightext.stbss.partinum[rstid])
            if oper.isferm:
                _sign = 1. if _partnum % 2 == 0 else -1.
                mat[lidx, ridx] = _sign * oper.mat[lbkid, rbkid]
            else:
                mat[lidx, ridx] = oper.mat[lbkid, rbkid]
    return BaseOperator(oper.siteidx, rightext, oper.isferm, mat, spin=oper.spin)


def rightsite_extend_oper(
        rightext: RightBlockExtend,
        oper: BaseOperator
    ):
    '''将RightBlockExtend中的site扩展到RightBlockExtend中'''
    opdim = oper.basis.dim
    if rightext.stbss.dim != opdim:
        raise ValueError('oper.basis.dim和RightBlockExtend.stbss.dim对不上')
    #这个时候是没有交换的符号的问题的，因为site在前面
    mat = numpy.zeros([rightext.dim, rightext.dim])
    #这个时候rlbk在高位，只有他们相等时才有数值
    for idx in range(rightext.rblk.dim):
        mat[idx*opdim:(idx+1)*opdim, idx*opdim:(idx+1)*opdim] = oper.mat
    return BaseOperator(oper.siteidx, rightext, oper.isferm, mat, spin=oper.spin)


def update_rightblockextend_oper(
        newrbk: RightBlock,
        oper: BaseOperator,
        phival
    ):
    '''利用21.19左右的式子'''
    if not isinstance(phival, numpy.ndarray):
        raise ValueError('phival不是ndarray')
    mat = numpy.matmul(oper.mat, phival.transpose())
    mat = numpy.matmul(phival, mat)
    return BaseOperator(oper.siteidx, newrbk, oper.isferm, mat, spin=oper.spin)
