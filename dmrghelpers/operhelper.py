"""
简化算符的创建与更新
"""

from collections import namedtuple
import numpy
import scipy.sparse
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

SINGLE_SITE_NUMBER_UP = numpy.array(
    [
        [0., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 1.]
    ]
)

SINGLE_SITE_NUMBER_DOWN = numpy.array(
    [
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ]
)


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

    @staticmethod
    def create_numup():
        '''粒子数向上的算符'''
        return OPERTUP(
            isferm=False,
            spin=1,
            mat=SINGLE_SITE_NUMBER_UP
        )

    @staticmethod
    def create_numdown():
        '''粒子数向下的算符'''
        return OPERTUP(
            isferm=False,
            spin=-1,
            mat=SINGLE_SITE_NUMBER_DOWN
        )

    @staticmethod
    def create_by_name(name):
        '''通过名字产生opertup'''
        dic = {
            'cu': OperFactory.create_spinup,
            'cd': OperFactory.create_spindown,
            'nu': OperFactory.create_numup,
            'nd': OperFactory.create_numdown,
            'u': OperFactory.create_u
        }
        return dic[name]()


def create_operator_of_site(basis: SiteBasis, tup: OPERTUP):
    '''从一个site创建，spin = 1代表向上，spin=-1代表向下
    '''
    if len(basis.sites) > 1:
        raise NotImplementedError('多个格子的不去实现')
    stidx = basis.sites[0]
    mat = scipy.sparse.csr_matrix(tup.mat)
    return BaseOperator(stidx, basis, tup.isferm, mat, spin=tup.spin)


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
    #只有idx2相等的时候才能有数值
    mat = scipy.sparse.block_diag([oper.mat] * 4)
    return BaseOperator(oper.siteidx, leftext, oper.isferm, mat, spin=oper.spin)


def leftsite_extend_oper(
        leftext: LeftBlockExtend,
        oper: BaseOperator
    ):
    '''将LeftBlockExtend.stbss上面的算符扩展到LeftBlockExtend上
    这里利用的是21.15左右的公式，加上了反对易
    Issue#3： 优化速度
    '''
    opdim = oper.basis.dim
    if leftext.stbss.dim != opdim:
        raise ValueError('oper.basis.dim和LeftBlockExtend.stbss.dim对应不上')
    opermat = oper.mat.todok()\
        if scipy.sparse.isspmatrix_coo(oper.mat) else oper.mat
    #需要处理反对易的符号
    #|A_1,A_2,..A_n> = C_1C_2..C_n|0>
    #C_n|A_1,A_2..> = C_nC_1C_2..|0> = (-1)^a C_1C_2..C_n|0>
    #其中a的数量是phi^n-1中的粒子数，因为phi^n-1是和粒子数的共同本征态
    eyedim = leftext.lblk.dim
    #
    speye = None
    if oper.isferm:
        #如果是反对易的，统计block中的算符数目
        eyeval = []
        #eye(eyedim)
        #numpy.zeros([eyedim, eyedim])
        for idx in leftext.lblk.iter_idx():
            _pnum = leftext.lblk.spin_nums[idx]
            _partinum = numpy.sum(_pnum)
            eyeval.append(1.0 if _partinum % 2 == 0 else -1.0)
            #speye[idx, idx] = 1.0 if _partinum % 2 == 0 else -1.0
        speye = scipy.sparse.dia_matrix((eyeval, 0), shape=(eyedim, eyedim))
        #speye = speye.tocsr()
    else:
        speye = scipy.sparse.eye(eyedim)
    #
    block_arr = []
    for lidx in leftext.stbss.iter_idx():
        row = []
        block_arr.append(row)
        for ridx in leftext.stbss.iter_idx():
            if opermat[lidx, ridx] == 0:
                if lidx == ridx:
                    row.append(scipy.sparse.csr_matrix((eyedim, eyedim)))
                else:
                    row.append(None)
            else:
                row.append(speye.multiply(opermat[lidx, ridx]))
    mat = scipy.sparse.bmat(block_arr)
    return BaseOperator(oper.siteidx, leftext, oper.isferm, mat, spin=oper.spin)


def update_leftblockextend_oper(
        newlbk: LeftBlock,
        oper: BaseOperator,
        spphival
    ):
    '''利用21.19左右的式子'''
    #if not isinstance(phival, numpy.ndarray):
    #    raise ValueError('phival不是ndarray')
    if not scipy.sparse.issparse(spphival):
        raise ValueError('spphival不是稀疏矩阵')
    mat = oper.mat * spphival.transpose()
    mat = spphival * mat
    return BaseOperator(oper.siteidx, newlbk, oper.isferm, mat, spin=oper.spin)


def rightblock_extend_oper(
        rightext: RightBlockExtend,
        oper: BaseOperator
    ):
    '''把rightblock.rblk下的算符扩展到
    rightblockextend上面
    Issue#3：优化速度
    '''
    opdim = oper.basis.dim
    if rightext.rblk.dim != opdim:
        raise ValueError('oper.basis.dim和RightBlockExtend.rblk.dim对应不上')
    opermat = oper.mat.todok()\
        if scipy.sparse.isspmatrix_coo(oper.mat) else oper.mat
    #需要处理反对易的符号
    #|A_1,A_2,..A_n> = C_1C_2..C_n|0>
    #在Cm|phi^n_beta> = |phi^n_beta'>的情况下
    #Cm|s^n-1,phi^n_beta> = -C^n-1Cm|0, phi^n_beta>
    #所以在扩展rightblock中的算符的时候，要看n-1上面有几个粒子
    eyedim = rightext.stbss.dim
    speye = None
    if oper.isferm:
        #如果是反对易的，统计block中的算符数目
        eyevals = []
        for idx in rightext.stbss.iter_idx():
            _pnum = rightext.stbss.partinum[idx]
            _partinum = numpy.sum(_pnum)
            eyevals.append(1.0 if _partinum % 2 == 0 else -1.0)
        speye = scipy.sparse.dia_matrix((eyevals, 0), shape=(eyedim, eyedim))
    else:
        speye = scipy.sparse.eye(eyedim)
    speye = speye.tocsr()
    #
    block_arr = numpy.array([[None]*rightext.rblk.dim]*rightext.rblk.dim)
    idxllist, idxrlist = opermat.nonzero()
    for lidx, ridx in zip(idxllist, idxrlist):
        block_arr[lidx, ridx] = speye.multiply(opermat[lidx, ridx])
    for idx in range(rightext.rblk.dim):
        if block_arr[idx, idx] is None:
            block_arr[idx, idx] = scipy.sparse.dok_matrix((eyedim, eyedim))
    mat = scipy.sparse.bmat(block_arr)
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
    #这个时候rlbk在高位，只有他们相等时才有数值
    mat = scipy.sparse.block_diag([oper.mat] * rightext.rblk.dim)
    #
    return BaseOperator(oper.siteidx, rightext, oper.isferm, mat, spin=oper.spin)


def update_rightblockextend_oper(
        newrbk: RightBlock,
        oper: BaseOperator,
        spphival
    ):
    '''利用21.19左右的式子'''
    #if not isinstance(phival, numpy.ndarray):
    #    raise ValueError('phival不是ndarray')
    if not scipy.sparse.issparse(spphival):
        raise ValueError('spphival不是稀疏矩阵')
    mat = oper.mat * spphival.transpose()
    mat = spphival * mat
    return BaseOperator(oper.siteidx, newrbk, oper.isferm, mat, spin=oper.spin)
