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
        spphival
    ):
    '''升级leftblockextend基上的哈密顿量到leftblock上'''
    #if not isinstance(phival, numpy.ndarray):
    #    raise ValueError('phival不是ndarray')
    if not scipy.sparse.issparse(spphival):
        raise ValueError('spphival不是稀疏矩阵')
    mat = ham.mat * spphival.transpose()
    mat = spphival * mat
    return Hamiltonian(lblk, mat)


def extend_rightblock_hamiltonian(ham: Hamiltonian, rbkext: RightBlockExtend):
    '''把哈密顿量扩展到更大的基上'''
    hamdim = ham.basis.dim
    if hamdim != rbkext.rblk.dim:
        raise ValueError('ham.basis.dim不等于lbkext.lblk.dim')
    #
    hammat = ham.mat.todok()\
        if scipy.sparse.isspmatrix_coo(ham.mat) else ham.mat
    #
    speye = scipy.sparse.eye(rbkext.stbss.dim).tocsr()
    #
    block_arr = numpy.array([[None]*rbkext.rblk.dim]*rbkext.rblk.dim)
    idxllist, idxrlist = hammat.nonzero()
    for lidx, ridx in zip(idxllist, idxrlist):
        block_arr[lidx, ridx] = speye.multiply(hammat[lidx, ridx])
    for idx in range(rbkext.rblk.dim):
        if block_arr[idx, idx] is None:
            block_arr[idx, idx] = scipy.sparse.dok_matrix(
                (rbkext.stbss.dim, rbkext.stbss.dim))
    mat = scipy.sparse.bmat(block_arr)
    return Hamiltonian(rbkext, mat)


def update_rightblockextend_hamiltonian(
        rblk: RightBlock,
        ham: Hamiltonian,
        spphival
    ):
    '''升级rightblockextend基上的哈密顿量到rightblock上'''
    #if not isinstance(phival, numpy.ndarray):
    #    raise ValueError('phival不是ndarray')
    #
    if not scipy.sparse.issparse(spphival):
        raise ValueError('spphval不是稀疏矩阵')
    mat = ham.mat * spphival.transpose()
    mat = spphival * mat
    return Hamiltonian(rblk, mat)


def plus_two_hamiltonian(ham1 :Hamiltonian, ham2: Hamiltonian):
    '''把两个哈密顿量加在一起，生成一个新的哈密顿量'''
    if ham1.basis.dim != ham2.basis.dim:
        raise ValueError('不在同一个基上')
    mat = ham1.mat + ham2.mat
    return Hamiltonian(ham1.basis, mat)
