"""dmrg的初始化有关的功能"""


from typing import List
from lattice import BaseModel
from fermionic.block import RightBlock
from dmrghelpers.blockhelper import first_leftblock, first_rightblock
from dmrghelpers.blockhelper import extend_rightblock
from dmrghelpers.hamhelper import create_hamiltonian_of_site
from dmrghelpers.hamhelper import extend_rightblock_hamiltonian
from dmrghelpers.operhelper import create_operator_of_site, OperFactory
from dmrghelpers.operhelper import rightsite_extend_oper, rightblock_extend_oper
from .storages import DMRGConfig


def init_first_site(
        model: BaseModel,
        nrg_maxkeep: int,
    ):
    '''初始化第一个site'''
    conf = DMRGConfig(model, nrg_maxkeep)
    #
    left = first_leftblock(model.sites[0])
    #创建第一个格子上的哈密顿量，还有第一个格子的产生算符
    hamleft = create_hamiltonian_of_site(left.fock_basis, model.coef_u, 0)
    cup1 = create_operator_of_site(left.fock_basis, OperFactory.create_spinup())
    cdn1 = create_operator_of_site(left.fock_basis, OperFactory.create_spindown())
    #把现在的结果暂存到dconf
    conf.left_tmp_reset(model.sites[0], left, hamleft)
    conf.left_tmp_add_oper(cup1)
    conf.left_tmp_add_oper(cdn1)
    #
    right = first_rightblock(model.sites[-1])
    #创建最后一个格子上的哈密顿量，还有最后一个格子的产生算符
    hamright = create_hamiltonian_of_site(right.fock_basis, model.coef_u, 0)
    cuplast = create_operator_of_site(right.fock_basis, OperFactory.create_spinup())
    cdnlast = create_operator_of_site(right.fock_basis, OperFactory.create_spindown())
    #把右侧的结果也暂存到dconf
    conf.right_tmp_reset(model.sites[-1], right, hamright)
    conf.right_tmp_add_oper(cuplast)
    conf.right_tmp_add_oper(cdnlast)
    return conf


def prepare_rightblockextend(
        right: RightBlock,
        phi_idx: int,
        conf: DMRGConfig,
        newbonds: List[int],
        extoper_storage: List[int]
    ):
    '''把最右边的一个格子的扩展计算出来，这个扩展要包括新增的hopping\n
    以后要用来构成superblock
    '''
    #这个过程不会更新conf中的right_tmp
    rightstorage = conf.get_rightblock_storage(phi_idx)
    rightham = rightstorage.hamiltonian
    #扩展right
    rightext = extend_rightblock(right)
    #给conf中的rightext进行初始化
    conf.rightext_reset(phi_idx, rightext)
    #扩展哈密顿量
    rightham = extend_rightblock_hamiltonian(rightham, rightext)
    #把扩展以后的哈密顿量存下来
    conf.storage_rightext_ham(phi_idx, rightham)
    #把需要扩展的算符扩展
    maintain_opers = {}
    #首先考虑这次新加进来的两个
    newup = create_operator_of_site(rightext.stbss, OperFactory.create_spinup())
    newup = rightsite_extend_oper(rightext, newup)
    newdown = create_operator_of_site(rightext.stbss, OperFactory.create_spindown())
    newdown = rightsite_extend_oper(rightext, newdown)
    #先把他放到maintain_opers当中
    maintain_opers[rightext.stbss.sites[0]] = (newup, newdown)
    #然后把现在暂存的这些都扩展并保存到maintain_oper
    #中，这些可能会去要计算bond，还有可能会保存到rightext
    while len(rightstorage.oper_storage_list) > 0:
        stidx = rightstorage.oper_storage_list[0]
        #弹出这两个算符
        stup, stdown = rightstorage.pop_oper(stidx)
        #扩展这两个算符，然后放到maintain_opers
        stup = rightblock_extend_oper(rightext, stup)
        stdown = rightblock_extend_oper(rightext, stdown)
        maintain_opers[stidx] = (stup, stdown)
    #把需要保存到ext中的算符都保存下来
    for stidx in extoper_storage:
        conf.storage_rightext_oper(phi_idx, maintain_opers[stidx][0])
        conf.storage_rightext_oper(phi_idx, maintain_opers[stidx][1])
    #把新的bond添加进去
    #rightext中保存的是rightham的实例，在这时修改也是可以的
    for bstidx in newbonds:
        #自旋向上部分
        tar_up = maintain_opers[bstidx][0]
        rightham.add_hopping_term(tar_up, newup)
        #自旋向下部分
        tar_down = maintain_opers[bstidx][1]
        rightham.add_hopping_term(tar_down, newdown)
    #把新的U添加进去
    newu = create_operator_of_site(rightext.stbss, OperFactory.create_u())
    rightham.add_u_term(newu, )
    return rightext
