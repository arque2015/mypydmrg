"""计算一维的ladder"""

from lattice.one_dim_ladder import HubbardLadder
from dmrg.dmrg_init import init_first_site, prepare_rightblockextend
from dmrg.nrg_iter import leftblock_to_next
from dmrg.dmrg_right_to_left import rightblockextend_to_next
from dmrg.dmrg_left_to_right import leftblockextend_to_next


def main():
    '''开始计算'''
    #晶格的配置
    lenx = 2
    modelsize = 3 * lenx
    spin_sector = (modelsize // 2, modelsize // 2)
    hubbard = HubbardLadder(lenx, 0.0, 0.05)
    #ED算基态能量的结果
    #U=0时结果是-12.63841536，U=1时是-11.13668625
    #U=2时结果是-9.78120910，U=4时是-7.52100677
    #print(hubbard)
    #for idx in range(1, modelsize+1):
    #    for bnd in hubbard.get_site_bonds(idx):
    #        print(idx, bnd, hubbard.get_t_coef(idx, bnd))
    #print(hubbard.get_t_coef(1, 5))
    #这个init_first_site创建DMRGConfig，同时把第一个格子和最后一个
    #格子的哈密顿量和算符存储到DMRGConfig中
    dconf = init_first_site(hubbard, 15)
    #TODO: DMRG过程中的裁减率以后应该实现，按照数量裁减误差不容易控制
    #DMRG时的maxkeep
    #left sweep的时候phi_idx从2...到modelsize - 3
    #这个phi_idx是ext的编号，所以共有modelsize-4次从ext升级到下一个ext
    #的过程，这个过程中，会从ext升级到block，保存多少个基，由dkeep指定
    dkeep = [20 + modelsize*idx for idx in range(modelsize-4)]
    #
    #开始warm up, nrg_iter是从一个leftblock到下一个leftblock
    #这个过程中，把leftblockext生成成功并且保存到DMRGConfig
    #因为需要modelsize-3位置上的ext，所以要一直更新到modelsize-2的block
    for phi_idx in range(2, modelsize-1):
        #这个phi_idx是目标block的编号
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
    #在warm up结束了以后，model size-3的leftext就有了，但是这个时候
    #还没有modesize上的rightext，先把这个位置的ext算出来
    extsite = modelsize - 1#新加的site
    #这个newbonds是N-1这个site和N这个site的bond
    newbonds = [bond for bond in\
        hubbard.get_site_bonds(extsite) if bond > extsite]
    #以后需要用的算符，这个里面主要是和leftext之间会有链接的
    site_need_tmp = []
    for stidx in range(extsite, modelsize+1):
        for bnd in hubbard.get_site_bonds(stidx):
            if bnd < extsite:
                site_need_tmp.append(stidx)
                break
    #准备第一个rightblockextend，这个过程把N上面的rightext保存下来
    right = dconf.get_rightblock_storage(modelsize).block
    _ = prepare_rightblockextend(right, modelsize, dconf, newbonds, site_need_tmp)
    #现在可以开始sweep了
    for _ in range(2):#计算两个sweep
        #开始right sweep
        #这个right sweep的时候，现在只有N上面的ext,从N-1的位置开始
        #推进ext，一直到4上的ext（4上的ext包含到第3个格子，range不包含最后一个）
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
            #从phi_idx+1推进到phi_idx
            gerg = rightblockextend_to_next(
                dconf, phi_idx, extrabonds, newbonds,
                site_need_tmp, spin_sector, dkeep[iter_idx]
            )
            print(gerg)
        #
        #开始left sweep
        #这次从2到N-3
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
