"""
创建Left/Right-Block,Left/Right-BlockExtend
update各种Block
"""

import numpy
from fermionic import DEBUG_MODE
from fermionic.block import LeftBlock, RightBlock
from fermionic.block import LeftBlockExtend, RightBlockExtend
from .sitehelper import site_of_idx, STATE_PARTINUM

def first_leftblock(startidx):
    '''创建第一个leftblock'''
    sbs = site_of_idx(startidx)
    initmat = None
    if DEBUG_MODE:
        initmat = numpy.eye(4)
    return LeftBlock(sbs, 4, initmat=initmat, partnum=STATE_PARTINUM)

def first_rightblock(stopidx):
    '''创建第一个rightblock'''
    sbs = site_of_idx(stopidx)
    initmat = None
    if DEBUG_MODE:
        initmat = numpy.eye(4)
    return RightBlock(sbs, 4, initmat=initmat, partnum=STATE_PARTINUM)

def extend_leftblock(lblk: LeftBlock):
    '''根据已经有的leftblock，扩展一个格子'''
    sites = lblk.fock_basis.sites
    newsiteidx = sites[-1] + 1
    newsite = site_of_idx(newsiteidx)
    return lblk.rdirect_product(newsite)

def extend_rightblock(rblk: RightBlock):
    '''根据已经有的rightblock，扩展一个格子'''
    sites = rblk.fock_basis.sites
    newsiteidx = sites[0] - 1
    newsite = site_of_idx(newsiteidx)
    return rblk.ldirect_product(newsite)

def update_to_leftblock(lbkext: LeftBlockExtend, phival, skipcheck=False):
    '''将一组phi带入21.6式，这个phi应该是通过求密度矩阵本正态得到的
    skipcheck=False的时候是要检查粒子数守恒的
    ``````
    TODO: 是否应该采用稀疏的phival保存方式节约内存？
    '''
    #对应phi值的alpha
    newstanum = len(phival)
    pnum_list = []
    #
    for alpha in range(newstanum):
        pnum = None
        #检查所有的pnum是否一样
        for rhsidx in lbkext.iter_idx():
            if phival[alpha, rhsidx] == 0:
                continue
            _pnum = lbkext.spin_nums[rhsidx]
            #apri, s_n = lbkext.idx_to_idxpair(rhsidx)
            #_pnum = lbkext.lblk.spin_nums[apri] + lbkext.stbss.partinum[s_n]
            if pnum is None:
                pnum = _pnum
                if skipcheck:#跳过检查其他的数值
                    break
            else:
                if _pnum[0] != pnum[0] or _pnum[1] != pnum[1]:
                    raise ValueError('update_to_leftblock: 粒子数不守恒')
        if pnum is None:
            raise ValueError('第%d个没有分量' % alpha)
        pnum_list.append(pnum)
    return lbkext.merge_to_block(phival, pnum_list=pnum_list)


def update_to_rightblock(rbkext: RightBlockExtend, phival, skipcheck=False):
    '''将一组phi带入21.6式，这个phi应该是通过求密度矩阵本正态得到的
    skipcheck=False时候是要检查粒子数守恒的,
    ``````
    TODO: 是否应该采用稀疏的phival保存方式节约内存？
    这个过程个update_to_leftblock是完全一样的，要不要合并？'''
    #对应phi值的alpha
    newstanum = len(phival)
    pnum_list = []
    for beta in range(newstanum):
        pnum = None
        #检查所有的pnum是否一样
        for rhsidx in rbkext.iter_idx():
            if phival[beta, rhsidx] == 0:
                continue
            _pnum = rbkext.spin_nums[rhsidx]
            #apri, s_n = lbkext.idx_to_idxpair(rhsidx)
            #_pnum = lbkext.lblk.spin_nums[apri] + lbkext.stbss.partinum[s_n]
            if pnum is None:
                pnum = _pnum
                if skipcheck:#跳过检查其他的数值
                    break
            else:
                if _pnum[0] != pnum[0] or _pnum[1] != pnum[1]:
                    raise ValueError('update_to_leftblock: 粒子数不守恒')
        if pnum is None:
            raise ValueError('第%d个没有分量' % beta)
        pnum_list.append(pnum)
    return rbkext.merge_to_block(phival, pnum_list=pnum_list)
