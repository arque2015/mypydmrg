"""计算一维的Hubbard链"""

from lattice.one_dim_chain import HubbardChain
from dmrg.dmrg_init import init_first_site, prepare_rightblockextend
from dmrg.nrg_iter import leftblock_to_next
from dmrg.dmrg_right_to_left import rightblockextend_to_next
from dmrg.dmrg_left_to_right import leftblockextend_to_next


def main():
    '''开始算法'''
    #首先设置格子
    #PBC时，6个格子基态能量-8.0，8个格子-9.65685425
    modelsize = 6
    hubbard = HubbardChain(modelsize)
    spin_sector = (modelsize // 2, modelsize // 2)
    #然后进行dmrg算法参数的配置
    dconf = init_first_site(hubbard, 15)
    #TODO: DMRG过程中的裁减率以后应该实现，按照数量裁减误差不容易控制
    #DMRG时的maxkeep
    dkeep = [15 + modelsize*idx for idx in range(modelsize-4)]
    #开始warm up，nrg_iter的时候是从一个leftblock到另一个leftblock
    #所以要做到最右边减2，这个时候有个多余的计算，就是计算了最后一个的升级
    for phi_idx in range(2, modelsize-1):
        #先拿到推进前的上一个leftblock
        leftstorage = dconf.get_leftblock_storage(phi_idx-1)
        leftblock = leftstorage.block
        #推进的时候会新加一个site，这个site就在phi_idx上，
        #找到这个site和已经有的site之间的bond
        newbonds = [bnd for bnd in hubbard.get_site_bonds(phi_idx) if bnd < phi_idx]
        #以后需要用到的算符这次也要存下来
        site_need_tmp = []
        for stidx in range(1, phi_idx+1):
            for bnd in hubbard.get_site_bonds(stidx):
                if bnd > phi_idx:
                    site_need_tmp.append(stidx)
                    break
        #推进到下一个leftblock
        leftblock_to_next(
            leftblock,
            phi_idx-1,
            dconf,
            newbonds,
            site_need_tmp,
            site_need_tmp
        )
    #warm up完成
    #把第一个rightblockextend做出来，用来做第一个superblock
    extsite = modelsize - 1#新加的site
    newbonds = [bond for bond in\
        hubbard.get_site_bonds(extsite) if bond > extsite]
    #以后需要用的算符
    site_need_tmp = []
    for stidx in range(extsite, modelsize+1):
        for bnd in hubbard.get_site_bonds(stidx):
            if bnd < extsite:
                site_need_tmp.append(stidx)
                break
    #准备第一个rightblockextend
    right = dconf.get_rightblock_storage(modelsize).block
    _ = prepare_rightblockextend(right, modelsize, dconf, newbonds, site_need_tmp)
    #开始DMRG计算
    for _ in range(2):#计算两个sweep
        #开始right sweep
        for iter_idx, phi_idx in enumerate(range(modelsize-1, 3, -1), 0):
            #找到leftblockextend和rightblockextend之间存在的bond
            #现在组成的superblock是phi^phi_idx-2,s^phi_idx-1,s^phi_idx,phi^phi_idx+1
            extrabonds = []
            leftext_edge = phi_idx-1
            for idx in range(1, leftext_edge + 1):
                _bonds = hubbard.get_site_bonds(idx)
                for bnd in _bonds:
                    if bnd > leftext_edge:
                        extrabonds.append((idx, bnd))
            #这次会把rightblockextend[phi_idx+1]升级成rightblock[phi_idx]
            #然后再把rightblock[phi_idx]扩展成rightblockextend[phi_idx],
            #这次扩展会把site[phi_idx-1]添加到rightblock[phi_idx]，这里找到
            #site[phi_idx-1]包含的bond
            newbonds = [bnd for bnd in\
                hubbard.get_site_bonds(phi_idx-1) if bnd > phi_idx-1]
            #rightblockextend[phi_idx]需要包含一些以后用的到的算符
            site_need_tmp = []
            for stidx in range(phi_idx-1, modelsize+1):#新加的格子是phi_idx-1
                for bnd in hubbard.get_site_bonds(stidx):
                    if bnd < phi_idx-1:
                        site_need_tmp.append(stidx)
                        break
            #
            gerg = rightblockextend_to_next(
                dconf, phi_idx, extrabonds, newbonds,
                site_need_tmp, spin_sector, dkeep[iter_idx]
            )
            print(gerg)
        #开始left sweep
        for iter_idx, phi_idx in enumerate(range(2, modelsize-2), 0):
            #找到leftblockextend和rightblockextend之间存在的bond
            #现在组成的superblock是phi^phi_idx-1,s^phi_idx,s^phi_idx+1,phi^phi_idx+2
            extrabonds = []
            leftext_edge = phi_idx
            for idx in range(1, leftext_edge+1):
                _bonds = hubbard.get_site_bonds(idx)
                for bnd in _bonds:
                    if bnd > leftext_edge:
                        extrabonds.append((idx, bnd))
            #这个时候会从leftblockextend[phi_idx-1]升级到leftblock[phi_idx]
            #然后再扩展到leftblockextend[phi_idx]，这时会增加phi_idx+1位置上的site
            #找到这个时候新加到leftblockextend上的bond
            newbonds = [bnd for bnd in\
                hubbard.get_site_bonds(phi_idx+1) if bnd < phi_idx+1]
            #以后会用到的算符
            site_need_tmp = []
            for stidx in range(1, phi_idx+2):
                for bnd in hubbard.get_site_bonds(stidx):
                    if bnd > phi_idx+1:
                        site_need_tmp.append(stidx)
                        break
            #
            gerg = leftblockextend_to_next(
                dconf, phi_idx, extrabonds, newbonds,
                site_need_tmp, spin_sector, dkeep[iter_idx]
            )
            print(gerg)


if __name__ == "__main__":
    main()
