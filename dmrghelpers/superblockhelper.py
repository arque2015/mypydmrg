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
    '''把rightblockextend基上面的哈密顿量扩展到superblock上'''
    #原本的哈密顿量在|s^n+1, phi^N-(n+1)>上
    #现在要弄到 |phi^n-1, s^n, s^n+1, phi^N-(n+1)>上
    # H' = I X H，由于哈密顿量里面都是算符的二次项，而且right中的
    #编号都比左边要大，所以不会产生反对易的符号
    mat = numpy.zeros([sbext.dim, sbext.dim])
    #循环idx并且把idx转成idxtuple的速度比较慢，循环idxtuple速度要快一点
    for ridxcode in sbext.iter_sitecode():
        for lidxcode in sbext.iter_sitecode():
            ridx = sbext.sitecode_to_idx(ridxcode)
            lidx = sbext.sitecode_to_idx(lidxcode)
            rlbkid, rlstid, rrstid, rrbkid = ridxcode
            llbkid, llstid, lrstid, lrbkid = lidxcode
            #只有leftext相同时才有数值（I）
            if rlbkid != llbkid or rlstid != llstid:
                continue
            rbk_lidx = sbext.rightblockextend.idxpair_to_idx(lrstid, lrbkid)
            rbk_ridx = sbext.rightblockextend.idxpair_to_idx(rrstid, rrbkid)
            mat[lidx, ridx] = ham.mat[rbk_lidx, rbk_ridx]
    return Hamiltonian(sbext, mat)


def rightext_oper_to_superblock(
        sbext: SuperBlockExtend,
        oper: BaseOperator
    ):
    '''把rightblockextend基上面的算符扩展到superblock上'''
    #原本的算符在|s^n+1, phi^N-(n+1)>上
    #现在要弄到 |phi^n-1, s^n, s^n+1, phi^N-(n+1)>上
    # O' = I X O，这时的算符会经过phi^n-1和s^n中所有的产生算符
    #才能到|s^n+1, phi^N-(n+1)>上
    mat = numpy.zeros([sbext.dim, sbext.dim])
    #循环idx并且把idx转成idxtuple的速度比较慢，循环idxtuple速度要快一点
    for ridxcode in sbext.iter_sitecode():
        for lidxcode in sbext.iter_sitecode():
            ridx = sbext.sitecode_to_idx(ridxcode)
            lidx = sbext.sitecode_to_idx(lidxcode)
            rlbkid, rlstid, rrstid, rrbkid = ridxcode
            llbkid, llstid, lrstid, lrbkid = lidxcode
            #只有leftext相同时才有数值（I）
            if rlbkid != llbkid or rlstid != llstid:
                continue
            #O中的指标
            rbk_lidx = sbext.rightblockextend.idxpair_to_idx(lrstid, lrbkid)
            rbk_ridx = sbext.rightblockextend.idxpair_to_idx(rrstid, rrbkid)
            if oper.isferm:
                #左边一共的粒子数目
                leftextid = sbext.leftblockextend.idxpair_to_idx(rlbkid, rlstid)
                _pnum = sbext.leftblockextend.spin_nums[leftextid]
                _partinum = numpy.sum(_pnum)
                #交换的符号
                _sign = 1. if _partinum % 2 == 0 else -1.
                mat[lidx, ridx] = _sign * oper.mat[rbk_lidx, rbk_ridx]
            else:
                mat[lidx, ridx] = oper.mat[rbk_lidx, rbk_ridx]
    return BaseOperator(oper.siteidx, sbext, oper.isferm, mat, spin=oper.spin)