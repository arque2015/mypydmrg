"""
和NRG算法的Phi有关的函数
"""

from typing import List
import numpy
from fermionic.baseop import Hamiltonian
from fermionic.block import LeftBlockExtend, LeftBlock
from dmrghelpers.blockhelper import extend_leftblock, update_to_leftblock
from dmrghelpers.hamhelper import extend_leftblock_hamiltonian, update_leftblockextend_hamiltonian
from dmrghelpers.operhelper import leftblock_extend_oper, leftsite_extend_oper
from dmrghelpers.operhelper import OperFactory, create_operator_of_site
from dmrghelpers.operhelper import update_leftblockextend_oper
#try:
from .storages import DMRGConfig
#except ImportError:
#    from storages import DMRGConfig

def leftblock_to_next(
        left: LeftBlock,
        phi_idx: int,
        conf: DMRGConfig,
        newbonds: List[int],
        extoper_storage: List[int],
        tmpoper_storage: List[int]
    ):
    '''将leftblock向前推进一个格子\n
    left是需要推进的block\n
    phi_idx是这个left的idx，这个和block_len不一定是一个东西\n
    conf是程序运行时的句柄\n
    newbonds是新加的这个site和哪些site有bond（新的site是left.fock_basis.site[-1]+1）\n
    extoper_storage是指|phi^phi_idx, s^phi_idx+1>这个基上有哪些算符需要保存\n
    村下来的extoper以后要用到superblock上。\n
    tmpoper_storage是指在|phi^phi_idx+1>这个基上要临时存储的算符，用在下一次递推\n
    时的哈密顿量计算，以及下一次的迭代中需要保存的ext\n
    '''
    leftstorage = conf.get_leftblock_storage(phi_idx)
    hamleft = leftstorage.hamiltonian
    ### 开始处理leftext在leftext上面工作的内容
    #把left扩展一个格子
    leftext = extend_leftblock(left)
    #给leftext进行初始化，leftext[phi_idx]中存的就是leftblock[phi_idx]的extend
    conf.leftext_reset(phi_idx, leftext)
    #把哈密顿量扩展一个格子
    hamleft = extend_leftblock_hamiltonian(hamleft, leftext)
    #把扩展基上的哈密顿量存下来
    conf.storage_leftext_ham(phi_idx, hamleft)
    #存储需要进行扩展的算符
    maintain_opers = {}
    #创建新的格子上厄的两个算符并且扩展
    newup = create_operator_of_site(leftext.stbss, OperFactory.create_spinup())
    newup = leftsite_extend_oper(leftext, newup)
    newdown = create_operator_of_site(leftext.stbss, OperFactory.create_spindown())
    newdown = leftsite_extend_oper(leftext, newdown)
    #if leftext.stbss.sites[0] in tmpoper_storage:
    maintain_opers[leftext.stbss.sites[0]] = (newup, newdown)
    #把left_tmp中所有的算符扩展
    #注意这个时候leftstorage.oper_storage_list会被修改，
    #不能在这个上面进行循环
    while len(leftstorage.oper_storage_list) > 0:
    #for stidx in leftstorage.oper_storage_list:
        stidx = leftstorage.oper_storage_list[0]
        #把left上面的两个算符弹出
        stup, stdown = leftstorage.pop_oper(stidx)
        #将两个算符扩展
        stup = leftblock_extend_oper(leftext, stup)
        stdown = leftblock_extend_oper(leftext, stdown)
        maintain_opers[stidx] = (stup, stdown)
    #把需要保存到leftext中的算符保存下来
    for stidx in extoper_storage:
        conf.storage_leftext_oper(phi_idx, maintain_opers[stidx][0])
        conf.storage_leftext_oper(phi_idx, maintain_opers[stidx][1])
    #找到构成bond的算符，把新的hopping添加到哈密顿量
    for bstidx in newbonds:
        #自旋上部分
        tar_up = maintain_opers[bstidx][0]
        hamleft.add_hopping_term(newup, tar_up)
        #自旋下部分
        tar_down =  maintain_opers[bstidx][1]
        hamleft.add_hopping_term(newdown, tar_down)
    ### 开始从leftext升级到下一个left
    #调用get_phival_from_hamleft，得到能量本正值
    phival = get_phival_from_hamleft(hamleft, leftext, conf.nrg_max_keep)
    #给leftext升级成新的基，这个时候phival其实没什么用
    #但fermionic在DEBUG_MODE下会解出basis在fock_basis下的系数，这个是需要的
    newleft = update_to_leftblock(leftext, phival)
    #给哈密顿量进行更新
    hamleft = update_leftblockextend_hamiltonian(newleft, hamleft, phival)
    #给conf中的left_tmp重置，现在left_tmp应该保存phi_idx+1的哈密顿量和算符了
    conf.left_tmp_reset(phi_idx+1, newleft, hamleft)
    #给下一次运算需要保存的算符更新
    for stidx in tmpoper_storage:
        up_ext, down_ext = maintain_opers[stidx]
        up_upd = update_leftblockextend_oper(newleft, up_ext, phival)
        down_upd = update_leftblockextend_oper(newleft, down_ext, phival)
        conf.left_tmp_add_oper(up_upd)
        conf.left_tmp_add_oper(down_upd)
    return newleft


def get_phival_from_hamleft(
        hamleft: Hamiltonian,
        leftext: LeftBlockExtend,
        maxkeep,
        restrict_sector=None#限制对角化时候的sector
    ):
    '''通过对角化获得phival\n
    在对角化的时候是给所有的sector都做对角化的\n
    然后再重新拼装成新的phival
    '''
    #查看left1ext中粒子数相同的block
    #自旋上或者下的粒子可以有0到block_len个
    maxnum = leftext.block_len + 1
    eigpairs = {}
    _tot_vecs = 0
    #限制sector的list
    sector_idx_list = None if restrict_sector is None\
        else [sec[0]+sec[1]*maxnum for sec in restrict_sector]
    #
    for supnum in range(maxnum):
        for sdnnum in range(maxnum):
            #如果不在限制的sector里面就跳过
            sector_idx = supnum + sdnnum * maxnum
            if sector_idx_list is not None:
                if not sector_idx in sector_idx_list:
                    continue
            #获得相同自旋粒子数的基的编号
            sidxlist = leftext.get_spin_sector([supnum, sdnnum])
            #如果没有这个sector直接跳过
            if sidxlist is None:
                continue
            _tot_vecs += len(sidxlist)
            #从哈密顿量中抽出相同粒子数组成的block
            blockmat = hamleft.get_block(sidxlist)
            #对角化这个sector的哈密顿量，哈密顿量是不会改变自旋上下的数目的，可以块对角化
            eigvals, eigvecs = numpy.linalg.eigh(blockmat)
            #numpy的结果中，eigvecs的每一列是一个本正态
            for idx, eva in enumerate(eigvals, 0):
                #phival中需要的是再整个LeftBlockExtend基上面的
                #扩展过去
                extend_vec = numpy.zeros(leftext.dim)
                for idx2, sidx in enumerate(sidxlist, 0):
                    extend_vec[sidx] = eigvecs[idx2, idx]
                if eva in eigpairs:
                    eigpairs[eva].append(extend_vec)
                else:
                    eigpairs[eva] = [extend_vec]
    _maxkeep = maxkeep
    if maxkeep > _tot_vecs:
        _maxkeep = _tot_vecs
    #下一个block的分量数量就是_maxkeep个
    phival = numpy.zeros([_maxkeep, leftext.dim])
    #phival中每一行应该是一个本正值
    #给本正值排序，从小到大
    eigvals_sorted = numpy.sort(list(eigpairs.keys()))
    #print(eigvals_sorted)
    phirow = 0
    for eva in eigvals_sorted:
        for eve in eigpairs[eva]:
            phival[phirow, :] = eve
            phirow += 1
            if phirow >= _maxkeep:
                break
        if phirow >= _maxkeep:
            break
    #利用phival更新|phi^2>
    return phival
