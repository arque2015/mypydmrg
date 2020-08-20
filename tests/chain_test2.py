"""
验证一维链上面的DMRG
``````
可以直接看examples里面的代码\n
"""

import numpy
from lattice.one_dim_chain import HubbardChain
from dmrg.storages import DMRGConfig
from dmrg.dmrg_init import init_first_site, prepare_rightblockextend
from dmrg.nrg_iter import leftblock_to_next
from dmrg.dmrg_iter import get_superblock_ham


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
    leftblock_to_next(dconf, target_site, newbonds, site_need_tmp, site_need_tmp, [])


def main():
    '''开始测试'''
    hc6 = HubbardChain(6, 2.0)
    dconf = init_first_site(hc6, 100, [])
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
    newbonds = [bond for bond in hc6.get_site_bonds(5) if bond > 5]
    print(newbonds)
    site_need_tmp = []
    for stidx in range(5, hc6.size+1):
        for bnd in hc6.get_site_bonds(stidx):
            if bnd < 5:
                site_need_tmp.append(stidx)
                break
    print('site_need_tmp ', site_need_tmp)
    right = dconf.get_rightblock_storage(6).block
    rightext = prepare_rightblockextend(dconf, 6, newbonds, site_need_tmp, [])
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
    #print(dconf._leftext_storage[3])
    #print(dconf._rightext_storage[6])
    _, mat, superext = get_superblock_ham(
        dconf.model,
        dconf.get_leftext_storage(3),
        dconf.get_rightext_storage(6),
        (3, 3), extrabonds
    )
    eigvals = numpy.linalg.eigvalsh(mat)
    print(eigvals)
    #可以验证六个格子PBC半满基态能量-8.0，调节dconf中的maxkeep
    #在U=1时能量时-6.60115829，U=2时能量-5.40945685
    #在U=3时能量时-4.43335361，U=4时能量-3.66870618
    #最大64（大于没效果），因为super是从leftext^3上面建立的



if __name__ == "__main__":
    main()
