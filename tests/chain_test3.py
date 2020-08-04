"""
验证一维链上面的DMRG
"""

import numpy
from lattice.one_dim_chain import HubbardChain
from dmrg.storages import DMRGConfig
from dmrg.dmrg_init import init_first_site, prepare_rightblockextend
from dmrg.nrg_iter import leftblock_to_next
from dmrg.dmrg_right_to_left import rightblockextend_to_next
#from dmrg.dmrg_iter import get_superblock_ham, get_density_root
#from dmrg.dmrg_iter import get_right_density_in_sector, get_right_phival

def push_left(dconf: DMRGConfig, target_site):
    '''将dconf推进一个格子'''
    model = dconf.model
    left_storage = dconf.get_leftblock_storage(target_site-1)
    leftblock = left_storage.block
    #fock_basis = left_storage.hamiltonian.basis.fock_basis
    #
    allbonds = model.get_site_bonds(target_site)
    #只需要比现在小的bond，大的都还没有
    newbonds = [bond for bond in allbonds if bond < target_site]
    #如果之前的这些site中依旧和更大的site有bond
    #需要在left_tmp中保留他们的算符，在下一次中扩展，
    #以便在扩展到这个格子的时候进行运算
    site_need_tmp = []
    for site in range(1, target_site+1):
        allbonds = model.get_site_bonds(site)
        for bond in allbonds:
            #大于target的在以后会用到，target这次要用的上次push时保存了
            if bond > target_site:
                site_need_tmp.append(site)
                break
    print(site_need_tmp)
    #leftext中需要存的算符可以用来算一些关联函数，暂时不用设置
    leftblock_to_next(leftblock, target_site-1, dconf, newbonds, site_need_tmp, site_need_tmp)


def main():
    '''开始测试'''
    hc6 = HubbardChain(6)
    dconf = init_first_site(hc6, 10)
    print(dconf)
    for stidx in [2, 3, 4]:
        push_left(dconf, stidx)
    ##lst4 = dconf.get_leftblock_storage(4)
    ##ham4 = lst4.hamiltonian
    ##print(ham4)
    ##print(ham4.basis.dim)
    ##eigvs = numpy.linalg.eigvalsh(ham4.mat)
    ##print(numpy.sort(eigvs))
    #print(dconf)
    #初始化最右侧的rightext6
    #这个时候新加的site是5号
    newbonds = [bond for bond in hc6.get_site_bonds(5) if bond > 5]
    print(newbonds)
    #小于5号的格子需要加在rightext6上面
    site_need_tmp = []
    for stidx in range(5, hc6.size+1):
        for bnd in hc6.get_site_bonds(stidx):
            if bnd < 5:
                site_need_tmp.append(stidx)
                break
    print('site_need_tmp ', site_need_tmp)
    right = dconf.get_rightblock_storage(6).block
    rightext = prepare_rightblockextend(right, 6, dconf, newbonds, site_need_tmp)
    print(rightext)
    #print(dconf._rightext_storage[6])
    #查看新的bond
    extrabonds = []
    leftext_edge = 4
    for idx in range(1, leftext_edge + 1):
        _bonds = hc6.get_site_bonds(idx)
        for bnd in _bonds:
            if bnd > leftext_edge:
                extrabonds.append((idx, bnd))
    print(extrabonds)
    newbonds = [bnd for bnd in hc6.get_site_bonds(4) if bnd > 4]
    site_need_tmp = []
    for stidx in range(4, 7):#新加的第四个格子，一直到第6个格子
        for bnd in hc6.get_site_bonds(stidx):
            if bnd < 4:
                site_need_tmp.append(stidx)
                break
    print(site_need_tmp)
    #包含小于4的bond的site都应该保存，包括ext5扩展到ext4的时候，需要3-4的bond
    #所以4也应该保存
    gerg = rightblockextend_to_next(
        dconf, 5, extrabonds, newbonds, site_need_tmp, (3, 3), 20
    )
    print(gerg)
    print(dconf.get_rightext_storage(5))
    #现在准备推进到下一个格子
    extrabonds = []
    leftext_edge = 3
    for idx in range(1, leftext_edge + 1):
        _bonds = hc6.get_site_bonds(idx)
        for bnd in _bonds:
            if bnd > leftext_edge:
                extrabonds.append((idx, bnd))
    print(extrabonds)
    newbonds = [bnd for bnd in hc6.get_site_bonds(3) if bnd > 3]
    site_need_tmp = []
    for stidx in range(3, 7):#新加的第四个格子，一直到第6个格子
        for bnd in hc6.get_site_bonds(stidx):
            if bnd < 3:
                site_need_tmp.append(stidx)
                break
    print(site_need_tmp)
    gerg = rightblockextend_to_next(
        dconf, 4, extrabonds, newbonds, site_need_tmp, (3, 3), 20
    )
    print(gerg)
    print(dconf.get_rightext_storage(4))
    #可以验证六个格子PBC半满基态能量-8.0，调节dconf中的maxkeep
    #最大64（大于没效果），因为super是从leftext^3上面建立的
    

if __name__ == "__main__":
    main()
