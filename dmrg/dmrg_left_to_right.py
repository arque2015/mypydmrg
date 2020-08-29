"""
简化dmrg_iter中的调用\n
和dmrg_right_to_left差不多
"""

from typing import List, Tuple
import numpy
import scipy.sparse.linalg
from dmrghelpers.blockhelper import update_to_leftblock, extend_leftblock
from dmrghelpers.hamhelper import update_leftblockextend_hamiltonian, extend_leftblock_hamiltonian
from dmrghelpers.operhelper import update_leftblockextend_oper
from dmrghelpers.operhelper import leftblock_extend_oper, leftsite_extend_oper
from dmrghelpers.operhelper import create_operator_of_site, OperFactory
from .dmrg_iter import get_superblock_ham, get_density_root
from .dmrg_iter import get_density_in_sector, get_phival_from_density_sector
from .storages import DMRGConfig


def leftblockextend_to_next(
        conf: DMRGConfig,
        phi_idx: int,
        extrabonds: List[int],
        newbonds: List[int],
        extoper_storage: List[int],
        measure_storage: List[Tuple[str, int]],
        spin_sector,
        maxkeep: int
    ):
    '''在右侧的rightext迭代到位以后，开始左侧的迭代\n
    这个时候的phi_idx是新的leftblockext的idx，新加的site就是phi_idx+1，\n
    相对应的rightblockext的idx就是phi_idx+2
    '''
    leftstorage = conf.get_leftext_storage(phi_idx-1)
    rightstorage = conf.get_rightext_storage(phi_idx+2)
    #首先要把superblock拼出来
    sector_idxs, mat, superext = get_superblock_ham(
        conf.model, leftstorage, rightstorage, spin_sector, extrabonds
    )
    #把基态解出来
    eigvals, eigvecs = scipy.sparse.linalg.eigsh(mat, k=1, which='SA')
    #numpy.linalg.eigh(mat)
    ground = eigvecs[:, 0]
    ground_erg = eigvals[0]
    #把基态的信息保留下来
    conf.ground_vec = ground
    conf.ground_secidx = sector_idxs
    conf.ground_superext = superext
    #构造密度矩阵
    lidxs, ridxs, mat = get_density_root(#pylint: disable=unused-variable
        leftstorage,
        rightstorage,
        sector_idxs,
        ground
    )
    #这个mat行是phi^phi_idx-1,s^phi_idx，列是s^phi_idx+1，phi^phi_idx+2
    #lidxs是mat上的行到leftext的基的指标
    #利用矩阵乘法收缩掉列
    denmat = numpy.matmul(mat, mat.transpose())
    #把密度矩阵再分成相同粒子数在一起的小块
    #这时就不需要到superblock上面的idx了，需要从phi^phi_idx-1,s^phi_idx
    #到leftext的idx
    spsec_mat_dict = get_density_in_sector(
        leftstorage,
        lidxs,
        denmat
    )
    #获得更新基的phival
    #这个过程就是给每一个denmat的subblock对角化，注意phival最后会
    #放到整个leftext的基上面去
    phival = get_phival_from_density_sector(
        leftstorage,
        spsec_mat_dict,
        maxkeep
    )
    #现在给leftblockext升级成新的位置上面的ext
    #先从leftblkext升级到left + 1
    leftext = leftstorage.block
    newleftblk = update_to_leftblock(leftext, phival)
    #然后升级哈密顿量
    newham = update_leftblockextend_hamiltonian(
        newleftblk,
        leftstorage.hamiltonian,
        phival
    )
    #调整一下tmp中的idx，没有实际的价值
    conf.left_tmp_reset(phi_idx, newleftblk, None)
    #先把newleftblk上面的算符升级，然后再扩展
    maintain_dict = {}
    for extidx in leftstorage.oper_storage_list:
        extup = leftstorage.get_oper(extidx, 1)
        newup = update_leftblockextend_oper(newleftblk, extup, phival)
        extdn = leftstorage.get_oper(extidx, -1)
        newdn = update_leftblockextend_oper(newleftblk, extdn, phival)
        maintain_dict[extidx] = (newup, newdn)
    #完成phi_idx位置上的leftblock，开始扩展它到leftblockextend
    newleftext = extend_leftblock(newleftblk)
    #把哈密顿量也扩展
    newhamext = extend_leftblock_hamiltonian(newham, newleftext)
    #以后的观测需要的算符，从leftext[phi_idx-1]中拿到，再升级扩展
    #以后放到leftext[phi_idx]中
    ext_meaops = {}
    for prefix, stidx in measure_storage:
        if stidx == phi_idx + 1:#如果是新加的格子上的算符，不需要升级
            #新建一个观测用的算符
            meaop = create_operator_of_site(
                newleftext.stbss,
                OperFactory.create_by_name(prefix)
            )
            #扩展到leftext[phi_idx]上
            meaop = leftsite_extend_oper(newleftext, meaop)
        else:
            #得到meaop
            meaop = leftstorage.get_meas(prefix, stidx)
            #升级meaop到leftblock[phi_idx]
            meaop = update_leftblockextend_oper(newleftblk, meaop, phival)
            #扩展
            meaop = leftblock_extend_oper(newleftext, meaop)
        ext_meaops['%s,%d' % (prefix, stidx)] = meaop
    #把现在的缓存进去
    conf.leftext_reset(phi_idx, newleftext)
    conf.storage_leftext_ham(phi_idx, newhamext)
    #获得新加的格子的两个算符并且扩展
    newsiteup = create_operator_of_site(newleftext.stbss, OperFactory.create_spinup())
    newsitedn = create_operator_of_site(newleftext.stbss, OperFactory.create_spindown())
    newsiteup = leftsite_extend_oper(newleftext, newsiteup)
    newsitedn = leftsite_extend_oper(newleftext, newsitedn)
    #把之前存的升级到newleftblk上的算符也扩展了
    for idx in maintain_dict:
        extup, extdn = maintain_dict[idx]
        extup = leftblock_extend_oper(newleftext, extup)
        extdn = leftblock_extend_oper(newleftext, extdn)
        maintain_dict[idx] = (extup, extdn)
    #新加的site是phi_idx + 1
    maintain_dict[phi_idx+1] = (newsiteup, newsitedn)
    #把新的hopping项加进去
    for bnd in newbonds:
        coef_t = conf.model.get_t_coef(bnd, newsiteup.siteidx)
        newhamext.add_hopping_term(maintain_dict[bnd][0], newsiteup, coef_t)
        newhamext.add_hopping_term(maintain_dict[bnd][1], newsitedn, coef_t)
    #把新的格子的U项添加进去
    newiu = create_operator_of_site(newleftext.stbss, OperFactory.create_u())
    newiu = leftsite_extend_oper(newleftext, newiu)
    newhamext.add_u_term(newiu, conf.model.coef_u)
    #把新的格子的Mu项添加进去
    coef_mu = conf.model.get_coef_mu(phi_idx+1)
    if coef_mu != 0:
        newnu = create_operator_of_site(newleftext.stbss, OperFactory.create_numup())
        newnu = leftsite_extend_oper(newleftext, newnu)
        newhamext.add_mu_term(newnu, coef_mu)
        newnd = create_operator_of_site(newleftext.stbss, OperFactory.create_numdown())
        newnd = leftsite_extend_oper(newleftext, newnd)
        newhamext.add_mu_term(newnd, coef_mu)
    #保存需要保存的算符
    for extidx in extoper_storage:
        conf.storage_leftext_oper(phi_idx, maintain_dict[extidx][0])
        conf.storage_leftext_oper(phi_idx, maintain_dict[extidx][1])
    #把需要保存的观测用的算符保存到leftext[phi_idx]当中
    leftstor_phiidx = conf.get_leftext_storage(phi_idx)
    for prefix, stidx in measure_storage:
        meaop = ext_meaops['%s,%d' % (prefix, stidx)]
        leftstor_phiidx.storage_meas(prefix, meaop)
    return ground_erg
