"""dmrg的初始化有关的功能"""


from typing import List, Tuple
from lattice import BaseModel
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
        meass: List[Tuple[str, int]]
    ):
    '''初始化第一个site\n
    这个时候只有1和N上面的block，创建这两个block上面的升降算符\n
    还有以后观测要用的算符。
    meass中应该是[(1, 'sz'), (N, 'sz')](N是最后一个格子的编号)\n
    这样的内容
    '''
    conf = DMRGConfig(model, nrg_maxkeep)
    #
    left = first_leftblock(model.sites[0])
    #创建第一个格子上的哈密顿量，还有第一个格子的产生算符
    hamleft = create_hamiltonian_of_site(
        left.fock_basis,
        model.coef_u,
        model.get_coef_mu(model.sites[0])
    )
    cup1 = create_operator_of_site(left.fock_basis, OperFactory.create_spinup())
    cdn1 = create_operator_of_site(left.fock_basis, OperFactory.create_spindown())
    #把现在的结果暂存到dconf
    conf.left_tmp_reset(model.sites[0], left, hamleft)
    conf.left_tmp_add_oper(cup1)
    conf.left_tmp_add_oper(cdn1)
    #
    right = first_rightblock(model.sites[-1])
    #创建最后一个格子上的哈密顿量，还有最后一个格子的产生算符
    hamright = create_hamiltonian_of_site(
        right.fock_basis,
        model.coef_u,
        model.get_coef_mu(model.sites[-1])
    )
    cuplast = create_operator_of_site(right.fock_basis, OperFactory.create_spinup())
    cdnlast = create_operator_of_site(right.fock_basis, OperFactory.create_spindown())
    #把右侧的结果也暂存到dconf
    conf.right_tmp_reset(model.sites[-1], right, hamright)
    conf.right_tmp_add_oper(cuplast)
    conf.right_tmp_add_oper(cdnlast)
    #把block上面的算符存储到tmp里，以后需要的是ext上面的算符
    leftstor = conf.get_leftblock_storage(model.sites[0])
    rightstor = conf.get_rightblock_storage(model.sites[-1])
    for prefix, idx in meass:
        if idx == model.sites[0]:
            stor = leftstor
            basis = left
        elif idx == model.sites[-1]:
            stor = rightstor
            basis = right
        else:
            raise ValueError('这个算符不在第一个或最后一个格子')
        measop = create_operator_of_site(
            basis.fock_basis,
            OperFactory.create_by_name(prefix)
        )
        stor.storage_meas(prefix, measop)
    return conf


def prepare_rightblockextend(
        conf: DMRGConfig,
        phi_idx: int,
        newbonds: List[int],
        extoper_storage: List[int],
        measure_storage: List[Tuple[str, int]]
    ):
    '''把最右边的一个格子的扩展计算出来，这个扩展要包括新增的hopping\n
    以后要用来构成superblock
    '''
    #这个过程不会更新conf中的right_tmp
    rightstorage = conf.get_rightblock_storage(phi_idx)
    right = rightstorage.block
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
        coef_t = conf.model.get_t_coef(bstidx, newup.siteidx)
        #自旋向上部分
        tar_up = maintain_opers[bstidx][0]
        rightham.add_hopping_term(tar_up, newup, coef_t)
        #自旋向下部分
        tar_down = maintain_opers[bstidx][1]
        rightham.add_hopping_term(tar_down, newdown, coef_t)
    #把新的U添加进去
    newu = create_operator_of_site(rightext.stbss, OperFactory.create_u())
    newu = rightsite_extend_oper(rightext, newu)
    rightham.add_u_term(newu, conf.model.coef_u)
    #把新的Mu项添加进去
    coef_mu = conf.model.get_coef_mu(phi_idx-1)
    if coef_mu != 0:
        newnu = create_operator_of_site(rightext.stbss, OperFactory.create_numup())
        newnu = rightsite_extend_oper(rightext, newnu)
        rightham.add_mu_term(newnu, coef_mu)
        newnd = create_operator_of_site(rightext.stbss, OperFactory.create_numdown())
        newnd = rightsite_extend_oper(rightext, newnd)
        rightham.add_mu_term(newnd, coef_mu)
    #把扩展后的算符存储到rightext[N]上面，用来给以后的观测使用
    #之前的过程并没有调整right_tmp，直接用就可以了
    rightext_stor = conf.get_rightext_storage(phi_idx)
    for prefix, idx in measure_storage:
        if idx == phi_idx - 1:#如果是新加的格子，就新建这个算符
            meaop = create_operator_of_site(
                rightext.stbss,
                OperFactory.create_by_name(prefix)
            )
            meaop = rightsite_extend_oper(rightext, meaop)
        else:
            meaop = rightstorage.get_meas(prefix, idx)
            meaop = rightblock_extend_oper(rightext, meaop)
        rightext_stor.storage_meas(prefix, meaop)
    return rightext
