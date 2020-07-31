"""
简化哈密顿量的创建和更新
"""

import numpy
from fermionic.block import LeftBlockExtend, LeftBlock
from fermionic.block import RightBlockExtend, RightBlock
from fermionic.baseop import Hamiltonian


def create_hamiltonian_of_site(basis, coef_u, coef_mu):
    '''单个格子的时候肯定是没有t的'''
    mat = numpy.zeros([basis.dim, basis.dim])
    #先不实现u和mu
    return Hamiltonian(basis, mat)


def extend_leftblock_hamiltonian(ham: Hamiltonian, lbkext: LeftBlockExtend):
    '''把哈密顿量扩展到更大的基上'''
    hamdim = ham.basis.dim
    if hamdim != lbkext.lblk.dim:
        raise ValueError('ham.basis.dim不等于lbkext.lblk.dim')
    mat = numpy.zeros([lbkext.dim, lbkext.dim])
    for idx in range(lbkext.stbss.dim):
        mat[idx*hamdim:(idx+1)*hamdim, idx*hamdim:(idx+1)*hamdim] = ham.mat
    return Hamiltonian(lbkext, mat)


def update_leftblockextend_hamiltonian(
        lblk: LeftBlock,
        ham: Hamiltonian,
        phival
    ):
    '''升级leftblockextend基上的哈密顿量到leftblock上'''
    if not isinstance(phival, numpy.ndarray):
        raise ValueError('phival不是ndarray')
    mat = numpy.matmul(ham.mat, phival.transpose())
    mat = numpy.matmul(phival, mat)
    return Hamiltonian(lblk, mat)


def extend_rightblock_hamiltonian(ham: Hamiltonian, rbkext: RightBlockExtend):
    '''把哈密顿量扩展到更大的基上'''
    hamdim = ham.basis.dim
    if hamdim != rbkext.rblk.dim:
        raise ValueError('ham.basis.dim不等于lbkext.lblk.dim')
    mat = numpy.zeros([rbkext.dim, rbkext.dim])
    for ridx in range(rbkext.dim):
        for lidx in range(rbkext.dim):
            rstid, rbkid = rbkext.idx_to_idxpair(ridx)
            lstid, lbkid = rbkext.idx_to_idxpair(lidx)
            if lstid != rstid:
                continue
            mat[lidx, ridx] = ham.mat[lbkid, rbkid]
    return Hamiltonian(rbkext, mat)


def update_rightblockextend_hamiltonian(
        rblk: RightBlock,
        ham: Hamiltonian,
        phival
    ):
    '''升级rightblockextend基上的哈密顿量到rightblock上'''
    if not isinstance(phival, numpy.ndarray):
        raise ValueError('phival不是ndarray')
    mat = numpy.matmul(ham.mat, phival.transpose())
    mat = numpy.matmul(phival, mat)
    return Hamiltonian(rblk, mat)


def plus_two_hamiltonian(ham1 :Hamiltonian, ham2: Hamiltonian):
    '''把两个哈密顿量加在一起，生成一个新的哈密顿量'''
    if ham1.basis.dim != ham2.basis.dim:
        raise ValueError('不在同一个基上')
    mat = ham1.mat + ham2.mat
    return Hamiltonian(ham1.basis, mat)
