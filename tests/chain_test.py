"""
测试一维链的功能
``````
可以直接看examples里面的代码\n
"""

import numpy
from dmrg.storages import DMRGConfig
from dmrg.nrg_iter import leftblock_to_next
from lattice.one_dim_chain import HubbardChain
from dmrghelpers.blockhelper import first_leftblock
from dmrghelpers.hamhelper import create_hamiltonian_of_site
from dmrghelpers.operhelper import OperFactory, create_operator_of_site

def test1():
    '''测试一维链上面的NRG，DMRG'''
    hc6 = HubbardChain(6, 0.0)
    #print(hc6)
    dconf = DMRGConfig(hc6, 1000)
    #
    left1 = first_leftblock(1)
    #创建第一个格子上的哈密顿量，还有第一个格子的产生算符
    hamleft = create_hamiltonian_of_site(left1.fock_basis, hc6.coef_u, 0)
    cup1 = create_operator_of_site(left1.fock_basis, OperFactory.create_spinup())
    cdn1 = create_operator_of_site(left1.fock_basis, OperFactory.create_spindown())
    #把现在的结果暂存到dconf
    dconf.left_tmp_reset(1, left1, hamleft)
    dconf.left_tmp_add_oper(cup1)
    dconf.left_tmp_add_oper(cdn1)
    print(dconf)
    #新的格子的bonds，只需要比自己小的
    allbonds = hc6.get_site_bonds(2)
    newbonds = [bond for bond in allbonds if bond < 2]
    print(newbonds)
    #如果之前的这些site中依旧和更大的site有bond
    #需要在left_tmp中保留他们的算符，在下一次中扩展
    site_need_tmp = []
    for site in range(1, left1.block_len+2):
        allbonds = hc6.get_site_bonds(site)
        for bond in allbonds:
            if bond > left1.block_len + 1:
                site_need_tmp.append(site)
                break
    print(site_need_tmp)
    left2 = leftblock_to_next(dconf, 2, newbonds, [], site_need_tmp, [])
    print(dconf)
    #继续向下一个格子
    allbonds = hc6.get_site_bonds(3)
    newbonds = [bond for bond in allbonds if bond < 3]
    print(newbonds)
    site_need_tmp = []
    for site in range(1, left2.block_len+2):
        allbonds = hc6.get_site_bonds(site)
        for bond in allbonds:
            if bond > left2.block_len + 1:
                site_need_tmp.append(site)
                break
    print(site_need_tmp)
    ##加上一个2用来检测结果
    #left3 = leftblock_to_next(left2, 2, dconf, newbonds, [], site_need_tmp + [2])
    ##检查算符
    #left_tmp = dconf.get_leftblock_storage(3)
    #cu1 = left_tmp.get_oper(1, 1)
    #cd1 = left_tmp.get_oper(1, -1)
    #cu2 = left_tmp.get_oper(2, 1)
    #cd2 = left_tmp.get_oper(2, -1)
    #for ridx in range(left3.dim):
    #    for lidx in range(left3.dim):
    #        print('右侧的基')
    #        print(left3.idx_to_state(ridx))
    #        #在fock_basis上面的分量
    #        sitebss_arr = left3.fock_dict[ridx]
    #        template = ''
    #        for idx2 in range(left3.fock_basis.dim):
    #            if (idx2 % 4) == 0:
    #                template += '\n'
    #            template += '%.4f |%s>\t' %\
    #                (sitebss_arr[idx2], left3.fock_basis.idx_to_state(idx2))
    #        print(template)
    #        print('左侧的基')
    #        #再fock_basis上面的分量
    #        sitebss_arr = left3.fock_dict[lidx]
    #        template = ''
    #        for idx2 in range(left3.fock_basis.dim):
    #            if (idx2 % 4) == 0:
    #                template += '\n'
    #            template += '%.4f |%s>\t' %\
    #                (sitebss_arr[idx2], left3.fock_basis.idx_to_state(idx2))
    #        print(template)
    #        print('算符的数值')
    #        print(cu1.mat[lidx, ridx])
    #        print(cd1.mat[lidx, ridx])
    #        print(cu2.mat[lidx, ridx])
    #        print(cd2.mat[lidx, ridx])
    #
    left3 = leftblock_to_next(dconf, 3, newbonds, [], site_need_tmp, [])
    #继续向下一个格子
    allbonds = hc6.get_site_bonds(4)
    newbonds = [bond for bond in allbonds if bond < 4]
    print(newbonds)
    #newbonds = newbonds + [1]#PBC的时候加一个4-1之间的链接
    site_need_tmp = []
    for site in range(1, left3.block_len+2):
        allbonds = hc6.get_site_bonds(site)
        for bond in allbonds:
            if bond > left3.block_len + 1:
                site_need_tmp.append(site)
                break
    print(site_need_tmp)
    left4 = leftblock_to_next(dconf, 4, newbonds, [], site_need_tmp, [])
    print(left4)
    lst4 = dconf.get_leftblock_storage(4)
    ham4 = lst4.hamiltonian
    print(ham4)
    print(ham4.basis.dim)
    eigvs = numpy.linalg.eigvalsh(ham4.mat)
    print(numpy.sort(eigvs))
    #半满四个格子的时候的精确解是-4.47213595（OBC）可以调节max_keep来验证
    #半满四个格子的时候的精确解是-4.0(PBC)可以去掉之前的注释验证
    #注意这里的解是没有限制sector的，可以看ham4.basis第一个分量是不是半满
    print(ham4.basis.spin_nums[0])


def main():
    '''开始测试'''
    test1()

if __name__ == "__main__":
    main()
