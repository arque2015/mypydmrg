"""
简化有关superblock的调用
"""

import numpy
from fermionic.block import LeftBlockExtend, RightBlockExtend
from fermionic.superblock import SuperBlockExtend
from fermionic.baseop import Hamiltonian, BaseOperator

def extend_merge_to_superblock(
        lbkext: LeftBlockExtend,
        rbkext: RightBlockExtend
    ):
    '''将两个extend合并成superblock'''
    return SuperBlockExtend(lbkext, rbkext)


def leftext_hamiltonian_to_superblock(
        sbext: SuperBlockExtend,
        ham: Hamiltonian
    ):
    '''将leftblockextend基上的hamiltonian扩展到superblock基上'''
    #原本的哈密顿量在|phi^n-1, s^n>上
    #现在要弄到 |phi^n-1, s^n, s^n+1, phi^N-(n+1)>上
    # H' = H X I
    mat = numpy.zeros([sbext.dim, sbext.dim])
    #右边的是高位
    highbit = sbext.rightblockextend.dim
    lowbitlen = sbext.leftblockextend.dim
    if lowbitlen != ham.basis.dim:
        raise ValueError('算符的基不一致')
    for idx in range(highbit):
        mat[idx*lowbitlen:(idx+1)*lowbitlen,\
            idx*lowbitlen:(idx+1)*lowbitlen] =\
                ham.mat
    return Hamiltonian(sbext, mat)


def leftext_oper_to_superblock(
        sbext: SuperBlockExtend,
        oper: BaseOperator
    ):
    '''将leftblockextend上面的算符扩展到superblockextend'''
    #原本的算符在|phi^n-1, s^n>上面
    #现在增加到|phi^n-1, s^n, s^n+1, phi^N-(n+1)>
    #没有影响正常的算符顺序，不会有反对易的符号所以扩展的方式和哈密顿量是差不多的
    mat = numpy.zeros([sbext.dim, sbext.dim])
    #
    highbit = sbext.rightblockextend.dim
    lowbitlen = sbext.leftblockextend.dim
    if lowbitlen != oper.basis.dim:
        raise ValueError('算符的基不一致')
    for idx in range(highbit):
        mat[idx*lowbitlen:(idx+1)*lowbitlen,\
            idx*lowbitlen:(idx+1)*lowbitlen] =\
                oper.mat
    return BaseOperator(oper.siteidx, sbext, oper.isferm, mat, spin=oper.spin)


def rightext_hamiltonian_to_superblock(
        sbext: SuperBlockExtend,
        ham: Hamiltonian
    ):
    '''把rightblockextend基上面的哈密顿量扩展到superblock上\n
    Issue#2: 优化算法以提升速度
    '''
    #原本的哈密顿量在|s^n+1, phi^N-(n+1)>上
    #现在要弄到 |phi^n-1, s^n, s^n+1, phi^N-(n+1)>上
    # H' = I X H，由于哈密顿量里面都是算符的二次项，而且right中的
    #编号都比左边要大，所以不会产生反对易的符号
    mat = numpy.zeros([sbext.dim, sbext.dim])
    eyedim = sbext.leftblockextend.dim
    #循环rightext
    for ridx in sbext.rightblockextend.iter_idx():
        for lidx in sbext.rightblockextend.iter_idx():
            #每一个right上面的值，现在都变成一个单位矩阵
            mat[lidx*eyedim:(lidx+1)*eyedim, ridx*eyedim:(ridx+1)*eyedim] =\
                numpy.eye(eyedim) * ham.mat[lidx, ridx]
    return Hamiltonian(sbext, mat)


def rightext_oper_to_superblock(
        sbext: SuperBlockExtend,
        oper: BaseOperator
    ):
    '''把rightblockextend基上面的算符扩展到superblock上\n
    Issue#2: 优化算法以提升速度
    '''
    #原本的算符在|s^n+1, phi^N-(n+1)>上
    #现在要弄到 |phi^n-1, s^n, s^n+1, phi^N-(n+1)>上
    # O' = I X O，这时的算符会经过phi^n-1和s^n中所有的产生算符
    #才能到|s^n+1, phi^N-(n+1)>上
    mat = numpy.zeros([sbext.dim, sbext.dim])
    eyedim = sbext.leftblockextend.dim
    #首先需要统计左边一共的粒子数目，计算费米符号
    eye = None
    if oper.isferm:
        eye = numpy.zeros([eyedim, eyedim])
        for idx in sbext.leftblockextend.iter_idx():
            _pnum = sbext.leftblockextend.spin_nums[idx]
            _partinum = numpy.sum(_pnum)
            eye[idx, idx] = 1. if _partinum % 2 == 0 else -1.
    else:
        eye = numpy.eye(eyedim)
    #循环rightext
    for ridx in sbext.rightblockextend.iter_idx():
        for lidx in sbext.rightblockextend.iter_idx():
            #ridx上面的leftblockextend产生的符号在eye中计算清楚了
            mat[lidx*eyedim:(lidx+1)*eyedim, ridx*eyedim:(ridx+1)*eyedim] =\
                eye * oper.mat[lidx, ridx]
    return BaseOperator(oper.siteidx, sbext, oper.isferm, mat, spin=oper.spin)
