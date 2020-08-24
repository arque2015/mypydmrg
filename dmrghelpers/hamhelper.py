"""
简化哈密顿量的创建和更新
"""

import numpy
import scipy.sparse
from fermionic.block import LeftBlockExtend, LeftBlock
from fermionic.block import RightBlockExtend, RightBlock
from fermionic.baseop import Hamiltonian
from .operhelper import SINGLE_SITE_INTERACT_U


def create_hamiltonian_of_site(basis, coef_u, coef_mu):
    '''单个格子的时候肯定是没有t的'''
    #用稀疏矩阵换掉
    #先不实现mu
    mat = scipy.sparse.csr_matrix(coef_u * SINGLE_SITE_INTERACT_U)
    mat = mat.tocsr()
    return Hamiltonian(basis, mat)


def extend_leftblock_hamiltonian(ham: Hamiltonian, lbkext: LeftBlockExtend):
    '''把哈密顿量扩展到更大的基上'''
    hamdim = ham.basis.dim
    if hamdim != lbkext.lblk.dim:
        raise ValueError('ham.basis.dim不等于lbkext.lblk.dim')
    mat = scipy.sparse.block_diag([ham.mat] * lbkext.stbss.dim)
    return Hamiltonian(lbkext, mat)


def update_leftblockextend_hamiltonian(
        lblk: LeftBlock,
        ham: Hamiltonian,
        phival
    ):
    '''升级leftblockextend基上的哈密顿量到leftblock上'''
    if not isinstance(phival, numpy.ndarray):
        raise ValueError('phival不是ndarray')
    spphival = scipy.sparse.csr_matrix(phival)
    mat = ham.mat * spphival.transpose()
    mat = spphival * mat
    return Hamiltonian(lblk, mat)


def extend_rightblock_hamiltonian(ham: Hamiltonian, rbkext: RightBlockExtend):
    '''把哈密顿量扩展到更大的基上'''
    hamdim = ham.basis.dim
    if hamdim != rbkext.rblk.dim:
        raise ValueError('ham.basis.dim不等于lbkext.lblk.dim')
    #
    block_arr = []
    speye = scipy.sparse.eye(rbkext.stbss.dim)
    for lidx in rbkext.rblk.iter_idx():
        row = []
        block_arr.append(row)
        for ridx in rbkext.rblk.iter_idx():
            if ham.mat[lidx, ridx] == 0:
                #保证维度是正确的
                if lidx == ridx:
                    row.append(speye.multiply(0))
                else:
                    row.append(None)
            else:
                row.append(speye.multiply(ham.mat[lidx, ridx]))
    mat = scipy.sparse.bmat(block_arr)
    #
    return Hamiltonian(rbkext, mat)


def update_rightblockextend_hamiltonian(
        rblk: RightBlock,
        ham: Hamiltonian,
        phival
    ):
    '''升级rightblockextend基上的哈密顿量到rightblock上'''
    if not isinstance(phival, numpy.ndarray):
        raise ValueError('phival不是ndarray')
    #
    spphival = scipy.sparse.csr_matrix(phival)
    mat = ham.mat * spphival.transpose()
    mat = spphival * mat
    return Hamiltonian(rblk, mat)


def plus_two_hamiltonian(ham1 :Hamiltonian, ham2: Hamiltonian):
    '''把两个哈密顿量加在一起，生成一个新的哈密顿量'''
    if ham1.basis.dim != ham2.basis.dim:
        raise ValueError('不在同一个基上')
    mat = ham1.mat + ham2.mat
    return Hamiltonian(ham1.basis, mat)
