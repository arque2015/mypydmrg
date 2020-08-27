"""
简化dmrg_init中的调用\n
和NRG的warm up不一样的是，warm up是以leftblock为基础的，leftblockext\n
和warm up的推进关系不大，保存leftblockext上面算符的目的是为了以后的DMRG使用\n
在DMRG的步骤中，rightextend是基础，rightblock和推进是没有什么关系的，\n
但是为了计算rightextend，也需要把rightextend上面的算符升级到新的rightblock上面
"""


from typing import List, Tuple
import numpy
import scipy.sparse.linalg
from dmrghelpers.blockhelper import extend_rightblock, update_to_rightblock
from dmrghelpers.hamhelper import extend_rightblock_hamiltonian
from dmrghelpers.hamhelper import update_rightblockextend_hamiltonian
from dmrghelpers.operhelper import rightsite_extend_oper, rightblock_extend_oper
from dmrghelpers.operhelper import OperFactory, create_operator_of_site
from dmrghelpers.operhelper import update_rightblockextend_oper
from .storages import DMRGConfig
from .dmrg_iter import get_superblock_ham, get_density_root
from .dmrg_iter import get_density_in_sector, get_phival_from_density_sector

def rightblockextend_to_next(
        conf: DMRGConfig,
        phi_idx: int,
        extrabonds: List[int],
        newbonds: List[int],
        extoper_storage: List[int],
        measure_storage: List[Tuple[str, int]],
        spin_sector,
        maxkeep: int
    ):
    '''这时候的phi_idx是新的rightext的idx，leftext的idx就是他减去2\n
    spin_sector是在superblock上面的粒子数的要求\n
    extrabond在get_superblock_ham中要使用，而newbonds是值扩展
    '''
    #首先要将superblock得到，这个时候左边和右边之间的
    #extrabond也要考虑上
    leftstorage = conf.get_leftext_storage(phi_idx-2)
    rightstorage = conf.get_rightext_storage(phi_idx+1)
    #scrtor_idxs中保存的是满足粒子数要求的所有superblock上的基的idx
    #mat是这个sector上的矩阵
    sector_idxs, mat, superext = get_superblock_ham(
        conf.model, leftstorage, rightstorage, spin_sector, extrabonds
    )
    #将基态解出来
    eigvals, eigvecs = scipy.sparse.linalg.eigsh(mat, k=1, which='SA')
    #numpy.linalg.eigh(mat)
    ground = eigvecs[:, 0]
    ground_erg = eigvals[0]
    #把基态的信息保留下来
    conf.ground_vec = ground
    conf.ground_secidx = sector_idxs
    conf.ground_superext = superext
    #print('g', ground_erg)
    #从基态构造密度矩阵
    #这里需要sector_idx来判断有哪些位置上是有数值的
    #这个方法主要是将基态的C(alpha,s^j,s^j+1,beta)这个写成
    #M（a,b）,lidxs就是将a变成（alpha，s^j）的列表，
    #ridxs是b到（s^j+1,beta）的，这在以后计算phival的时候有作用
    #因为phival的列是在（s^j+1,beta）的
    lidxs, ridxs, mat = get_density_root(#pylint: disable=unused-variable
        leftstorage,
        rightstorage,
        sector_idxs,
        ground
    )
    #right上面的密度矩阵的获得
    denmat = numpy.matmul(mat.transpose(), mat)
    #这个方法主要目的是把density matrix划分成小块
    spsec_mat_dict = get_density_in_sector(
        rightstorage,
        ridxs,
        denmat
    )
    #获得phival
    phival = get_phival_from_density_sector(
        rightstorage,
        spsec_mat_dict,
        maxkeep
    )
    #在获得了phival以后，就可以通过这个phival更新基
    #rightblock^phi_idx里面的内容全部都在rightblockextend^phi_idx+1里面
    #不需要添加，只需要升级就行了，在DMRG sweep的过程中，rightblock上面的
    #算符或者哈密顿量都是用不到的，不进行保存
    rightext = rightstorage.block
    newrightblk = update_to_rightblock(rightext, phival)
    newham = update_rightblockextend_hamiltonian(
        newrightblk,
        rightstorage.hamiltonian,
        phival)
    #设置一下现在的idx，不保存哈密顿量，以后用不到
    conf.right_tmp_reset(phi_idx, newrightblk, None)
    #用来保存newrightblk上面的算符
    maintain_dict = {}
    #把需要放到新的rightext上面的算符，先升级到新的right上面
    for extidx in rightstorage.oper_storage_list:#extoper_storage:
        #上一次迭代的时候，包含小于phi_val的bond的site都被存在
        #conf里面了
        extup = rightstorage.get_oper(extidx, 1)
        newup = update_rightblockextend_oper(newrightblk, extup, phival)
        extdn = rightstorage.get_oper(extidx, -1)
        newdn = update_rightblockextend_oper(newrightblk, extdn, phival)
        maintain_dict[extidx] = (newup, newdn)
    #把新的rightblock扩展一个格子
    newrightext = extend_rightblock(newrightblk)
    #把哈密顿量也扩展一个格子
    newhamext = extend_rightblock_hamiltonian(newham, newrightext)
    #以后的观测需要的算符，从leftext[phi_idx-1]中拿到，再升级扩展
    #以后放到leftext[phi_idx]中
    ext_meaops = {}
    for prefix, stidx in measure_storage:
        if stidx == phi_idx - 1:#如果是新加的格子上的算符，不需要升级
            #新建一个观测用的算符
            meaop = create_operator_of_site(
                newrightext.stbss,
                OperFactory.create_measure(prefix)
            )
            #扩展到leftext[phi_idx]上
            meaop = rightsite_extend_oper(newrightext, meaop)
        else:
            #得到meaop
            meaop = rightstorage.get_meas(prefix, stidx)
            #升级meaop到leftblock[phi_idx]
            meaop = update_rightblockextend_oper(newrightblk, meaop, phival)
            #扩展
            meaop = rightblock_extend_oper(newrightext, meaop)
        ext_meaops['%s,%d' % (prefix, stidx)] = meaop
    #更新缓存的ext，现在存进去就行了，后面在add_hopping_term
    #也是加在这个newhamext上面的
    conf.rightext_reset(phi_idx, newrightext)
    conf.storage_rightext_ham(phi_idx, newhamext)
    #获得新加的格子上面的两个算符
    newsiteup = create_operator_of_site(newrightext.stbss, OperFactory.create_spinup())
    newsitedn = create_operator_of_site(newrightext.stbss, OperFactory.create_spindown())
    #扩展到新的基上
    newsiteup = rightsite_extend_oper(newrightext, newsiteup)
    newsitedn = rightsite_extend_oper(newrightext, newsitedn)
    #maintain_dict[phi_idx-1] = (newsiteup, newsitedn)
    #把maintain_dict中的其他算符也扩展到新的基上
    for idx in maintain_dict:
        extup, extdn = maintain_dict[idx]
        extup = rightblock_extend_oper(newrightext, extup)
        extdn = rightblock_extend_oper(newrightext, extdn)
        maintain_dict[idx] = (extup, extdn)
    maintain_dict[phi_idx-1] = (newsiteup, newsitedn)
    #加入新的hopping项
    for bnd in newbonds:
        coef_t = conf.model.get_t_coef(bnd, newsiteup.siteidx)
        newhamext.add_hopping_term(maintain_dict[bnd][0], newsiteup, coef_t)
        newhamext.add_hopping_term(maintain_dict[bnd][1], newsitedn, coef_t)
    #把新的格子的U项添加进去
    newiu = create_operator_of_site(newrightext.stbss, OperFactory.create_u())
    newiu = rightsite_extend_oper(newrightext, newiu)
    newhamext.add_u_term(newiu, conf.model.coef_u)
    #保存需要保存的新的ext上面的算符
    for extidx in extoper_storage:
        conf.storage_rightext_oper(phi_idx, maintain_dict[extidx][0])
        conf.storage_rightext_oper(phi_idx, maintain_dict[extidx][1])
    #把需要保存的观测用的算符保存到leftext[phi_idx]当中
    rightstor_phiidx = conf.get_rightext_storage(phi_idx)
    for prefix, stidx in measure_storage:
        meaop = ext_meaops['%s,%d' % (prefix, stidx)]
        rightstor_phiidx.storage_meas(prefix, meaop)
    return ground_erg
