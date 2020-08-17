"""
包含DMRG的sweep的函数
"""

import numpy
from dmrghelpers.superblockhelper import extend_merge_to_superblock
from dmrghelpers.superblockhelper import leftext_hamiltonian_to_superblock
from dmrghelpers.superblockhelper import rightext_hamiltonian_to_superblock
from dmrghelpers.superblockhelper import leftext_oper_to_superblock
from dmrghelpers.superblockhelper import rightext_oper_to_superblock
from dmrghelpers.hamhelper import plus_two_hamiltonian
from .storages import BlockStorage, DMRGConfig

def get_superblock_ham(
        conf: DMRGConfig,
        leftstorage: BlockStorage,
        rightstorage: BlockStorage,
        spin_sector,
        bonds
    ):
    '''获得superblock上的哈密顿量，这个时候需要指定sector\n
    '''
    leftext = leftstorage.block
    leftham = leftstorage.hamiltonian
    rightext = rightstorage.block
    rightham = rightstorage.hamiltonian
    superext = extend_merge_to_superblock(leftext, rightext)
    #print(superext)
    #把左边的哈密顿量扩展到superblock
    lefthamext = leftext_hamiltonian_to_superblock(superext, leftham)
    #把右边的哈密顿量扩展到superblock
    righthamext = rightext_hamiltonian_to_superblock(superext, rightham)
    #把两个结果加起来
    superham = plus_two_hamiltonian(lefthamext, righthamext)
    del lefthamext
    del righthamext
    #把bond加起来
    op_dict = {}
    for bond in bonds:
        op1_idx, op2_idx = bond
        #找出第一个位置上的两个算符
        if op1_idx in op_dict:
            op1up, op1down = op_dict[op1_idx]
        else:
            op1up = leftstorage.get_oper(op1_idx, 1)
            #op1up = leftext_oper_to_superblock(superext, op1up)
            op1down = leftstorage.get_oper(op1_idx, -1)
            #op1down = leftext_oper_to_superblock(superext, op1down)
            op_dict[op1_idx] = (op1up, op1down)
        #找出第二个位置上的两个算符
        if op2_idx in op_dict:
            op2up, op2down = op_dict[op2_idx]
        else:
            op2up = rightstorage.get_oper(op2_idx, 1)
            #op2up = rightext_oper_to_superblock(superext, op2up)
            op2down = rightstorage.get_oper(op2_idx, -1)
            #op2down = rightext_oper_to_superblock(superext, op2down)
            op_dict[op2_idx] = (op2up, op2down)
        #把这个新的hopping项加进去
        #Issue #16: 现在不需要扩展到superblock上的算符
        #只需要在ext上面的算符，op_dict以后也没有用了
        coef_t = conf.model.get_t_coef(op1_idx, op2_idx)
        superham.superblock_add_hopping_term(op1up, op2up, coef_t)
        superham.superblock_add_hopping_term(op1down, op2down, coef_t)
    #找到符合sector的所有idx
    sector_idxs = []
    for stcode in superext.iter_sitecode():
        lbkid, lstid, rstid, rbkid = stcode
        superidx = superext.sitecode_to_idx(stcode)
        lextidx = leftext.idxpair_to_idx(lbkid, lstid)
        rextidx = rightext.idxpair_to_idx(rstid, rbkid)
        nspin = leftext.spin_nums[lextidx] + rightext.spin_nums[rextidx]
        if nspin[0] == spin_sector[0] and nspin[1] == spin_sector[1]:
            sector_idxs.append(superidx)
    #
    #print('len', len(sector_idxs))
    mat = superham.get_block(sector_idxs)
    return sector_idxs, mat


def get_density_root(
        leftstorage: BlockStorage,
        rightstorage: BlockStorage,
        super_idxs,
        eigvec
    ):
    '''computational many particle physics第21.36\n
    将其右手边改造成矩阵乘法，这个方法用来构造右手边的矩阵\n
    对于right，他的转置和他相乘结果就是密度矩阵\n
    对于left，他乘以他的转置就是密度矩阵
    '''
    #把基态包含的leftext和rightext的idx拆开，
    #这些idx以后还是leftext和rightext基的idx，可以用来
    #判断密度矩阵的sector
    leftext = leftstorage.block
    rightext = rightstorage.block
    leftextidxs = []
    rightextidxs = []
    pnum_ = None
    for sidx in super_idxs:
        rextidx = numpy.floor_divide(sidx, leftext.dim)
        lextidx = sidx - rextidx * leftext.dim
        _pnum = leftext.spin_nums[lextidx] + rightext.spin_nums[rextidx]
        if pnum_ is None:
            pnum_ = _pnum
        else:
            if pnum_[0] != _pnum[0] or pnum_[1] != _pnum[1]:
                raise ValueError('本征态粒子数不守恒')
        if lextidx not in leftextidxs:
            leftextidxs.append(lextidx)
        if rextidx not in rightextidxs:
            rightextidxs.append(rextidx)
    #
    mat = numpy.zeros([len(leftextidxs), len(rightextidxs)])
    #注意这个矩阵的第n行是leftext中的第leftextidxs[n]个基
    #这个矩阵的第n列是rightext中的第rightextidxs[n]个基
    #而super_idxs中的第n和值是superblock中的第super_idx[n]个基
    for idx, sidx in enumerate(super_idxs, 0):
        rextidx = numpy.floor_divide(sidx, leftext.dim)
        lextidx = sidx - rextidx * leftext.dim
        lidx = leftextidxs.index(lextidx)
        ridx = rightextidxs.index(rextidx)
        mat[lidx, ridx] = eigvec[idx]
    return leftextidxs, rightextidxs, mat


def get_density_in_sector(
        storage: BlockStorage,
        extidxs,
        denmat
    ):
    '''这个方法主要是把denmat划分好block，这个和left或者right没有关系\n
    区别是denmat，在调用前应该配置好
    '''
    blkext = storage.block
    block_len = blkext.block_len
    max_sec_num = block_len + 1
    #先找出所有rightext上面有相同粒子数的sector
    #最后在形成phival的时候，所有的有相同sector的放在一起
    #rightextidxs这个数组联系起来了mat上的指标和
    #rightext的基的指标，给密度矩阵分好块以后，每一块
    #也有一个类似的数组，把每一块中的指标联系到rightext的基上面
    spin_sector_dict = {}
    for exidx in extidxs:#这里面是指向-BlockExtend的数值
        spin_nums = blkext.spin_nums[exidx]
        spidx = spin_nums[1] * max_sec_num + spin_nums[0]
        if spidx in spin_sector_dict:
            spin_sector_dict[spidx].append(exidx)
        else:
            spin_sector_dict[spidx] = [exidx]
    #所有具有相同的粒子数的-BlockExtend编号都放在了一起
    #检查密度矩阵的大小
    shape = numpy.shape(denmat)
    if shape[0] != len(extidxs) or shape[1] != len(extidxs):
        raise ValueError('密度矩阵大小不对')
    #denmat = numpy.matmul(mat.transpose(), mat)
    #找出每个sector对应的矩阵的块
    spin_sector_mat_dict = {}
    #
    for secidx in spin_sector_dict:
        ssecidxs = spin_sector_dict[secidx]
        #denmat中的一个subblock
        mat = numpy.zeros([len(ssecidxs), len(ssecidxs)])
        #组织这个subblock的顺序：这个subblock中第n个在-ExtendBlock中是第ssecidxs[n]
        for lidx, lextidx in enumerate(ssecidxs, 0):
            for ridx, rextidx in enumerate(ssecidxs, 0):
                #denmat中的指标是第n个是right中第rightextidxs[n]个基
                #lextidx和rextidx也是rightext上面基的编号
                #首先要找到他在denmat中的编号
                dlidx = extidxs.index(lextidx)
                dridx = extidxs.index(rextidx)
                mat[lidx, ridx] = denmat[dlidx, dridx]
        spin_sector_mat_dict[secidx] = (ssecidxs, mat)
        #查看其他block是不是0
        #for secidx2 in spin_sector_dict:
        #    if secidx2 == secidx:
        #        continue
        #    ssecidxs2 = spin_sector_dict[secidx2]
        #    mat = numpy.zeros([len(ssecidxs), len(ssecidxs2)])
        #    for lidx, lextidx in enumerate(ssecidxs, 0):
        #        for ridx, rextidx in enumerate(ssecidxs2, 0):
        #            dlidx = extidxs.index(lextidx)
        #            dridx = extidxs.index(rextidx)
        #            mat[lidx, ridx] = denmat[dlidx, dridx]
        #    print('mat sum: ', numpy.allclose(mat, 0))
    return spin_sector_mat_dict


def get_phival_from_density_sector(
        storage: BlockStorage,
        sp_sec_mat_dict,
        maxkeep
    ):
    '''得到更新rightext的phival'''
    extblk = storage.block
    #
    eigpair = {}
    _tot_vec = 0
    #给每个sector的密度矩阵对角化
    for sector in sp_sec_mat_dict:
        rsecidxs, mat = sp_sec_mat_dict[sector]
        eigvals, eigvecs = numpy.linalg.eigh(mat)
        _tot_vec += len(rsecidxs)
        for idx, eva in enumerate(eigvals, 0):
            #这个时候，要把sector上面的向量扩展到整个rightext基上
            extend_vec = numpy.zeros(extblk.dim)
            for idx2, bidx in enumerate(rsecidxs, 0):
                extend_vec[bidx] = eigvecs[idx2, idx]
            if eva in eigpair:
                eigpair[eva].append(extend_vec)
            else:
                eigpair[eva] = [extend_vec]
    #
    _maxkeep = maxkeep
    if _maxkeep > _tot_vec:
        _maxkeep = _tot_vec
    #选择前_maxkeep个分量
    phival = numpy.zeros([_maxkeep, extblk.dim])
    #给本正值排序
    #和NRG给能量本正值排序不同，这里降序排序
    eigvals_sorted = numpy.sort(list(eigpair.keys()))[::-1]
    #验证密度矩阵本正值的求和
    #densum = 0
    #for eva in eigvals_sorted:
    #    densum += len(eigpair[eva]) * eva
    #print('density sum: ', densum)
    #把前_max_keep个赋值给phi_val
    phirow = 0
    densum = 0
    for eva in eigvals_sorted:
        for eve in eigpair[eva]:
            phival[phirow, :] = eve
            phirow += 1
            densum += eva
            if phirow >= _maxkeep:
                break
        if phirow >= _maxkeep:
            break
    print('density remain: ', densum)
    #
    return phival
