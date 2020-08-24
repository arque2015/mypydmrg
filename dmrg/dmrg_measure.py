"""简化观测的时候的调用"""

import numpy
import scipy.sparse
from fermionic.baseop import BaseOperator
from dmrghelpers.superblockhelper import leftext_oper_to_superblock
from dmrghelpers.superblockhelper import rightext_oper_to_superblock
from dmrghelpers.superblockhelper import extend_merge_to_superblock
from .storages import DMRGConfig


def measure_oper_of_site(conf: DMRGConfig, prefix: str, stidx: int):
    '''观测一个格子上的某个算符'''
    leftext = conf.ground_superext.leftblockextend
    rightext = conf.ground_superext.rightblockextend
    superext = extend_merge_to_superblock(leftext, rightext)
    #
    leftphi = leftext.block_len - 1
    rightphi = conf.model.size - rightext.block_len + 2
    #
    leftstorage = conf.get_leftext_storage(leftphi)
    rightstorage = conf.get_rightext_storage(rightphi)
    #return leftstorage, rightstorage
    if stidx < leftphi + 2:
        measoper = leftstorage.get_meas(prefix, stidx)
        measoper = leftext_oper_to_superblock(superext, measoper)
        measmat = measoper.get_block(conf.ground_secidx)
    elif stidx > rightphi - 2:
        measoper = rightstorage.get_meas(prefix, stidx)
        measoper = rightext_oper_to_superblock(superext, measoper)
        measmat = measoper.get_block(conf.ground_secidx)
    else:
        raise ValueError('没有这个指标')
    #val = numpy.matmul(conf.ground_vec,\
    #    numpy.matmul(measmat.toarray(), conf.ground_vec))
    spground = numpy.expand_dims(conf.ground_vec, 1)
    spground = scipy.sparse.csr_matrix(spground)
    val = measmat * spground
    val = spground.transpose() * val
    val = val[0, 0]
    return val


def measure_corr_of_2sites(
        conf: DMRGConfig,
        prefix: str,
        stidx1: int, stidx2: int
    ):
    '''观测两个格子上的关联'''
    leftext = conf.ground_superext.leftblockextend
    rightext = conf.ground_superext.rightblockextend
    superext = extend_merge_to_superblock(leftext, rightext)
    #
    leftphi = leftext.block_len - 1
    rightphi = conf.model.size - rightext.block_len + 2
    #
    leftstorage = conf.get_leftext_storage(leftphi)
    rightstorage = conf.get_rightext_storage(rightphi)
    #
    if stidx1 < leftphi + 2:
        measoper1 = leftstorage.get_meas(prefix, stidx1)
        measoper1 = leftext_oper_to_superblock(superext, measoper1)
    elif stidx1 > rightphi - 2:
        measoper1 = rightstorage.get_meas(prefix, stidx1)
        measoper1 = rightext_oper_to_superblock(superext, measoper1)
    else:
        raise ValueError('stidx1的指标没有')
    #
    if stidx2 < leftphi + 2:
        measoper2 = leftstorage.get_meas(prefix, stidx2)
        measoper2 = leftext_oper_to_superblock(superext, measoper2)
    elif stidx2 > rightphi - 2:
        measoper2 = rightstorage.get_meas(prefix, stidx2)
        measoper2 = rightext_oper_to_superblock(superext, measoper2)
    #
    coropmat = measoper1.mat * measoper2.mat
    #numpy.matmul(measoper1.mat.toarray(), measoper2.mat.toarray())
    #coropmat = coropmat[conf.ground_secidx]
    #coropmat = coropmat[:, conf.ground_secidx]
    coropmat = coropmat.tocsr()[conf.ground_secidx]
    coropmat = coropmat.tocsc()[:,conf.ground_secidx]
    #
    #val = numpy.matmul(conf.ground_vec,\
    #    numpy.matmul(coropmat, conf.ground_vec))
    spground = numpy.expand_dims(conf.ground_vec, 1)
    spground = scipy.sparse.csr_matrix(spground)
    val = coropmat * spground
    val = spground.transpose() * val
    val = val[0, 0]
    return val
