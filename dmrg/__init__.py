"""
Numerical Renormalization Group算法有关的功能
主要用来做DMRG算法的warm up
"""

from typing import Tuple, List
from lattice import BaseModel
from dmrg.dmrg_init import init_first_site, prepare_rightblockextend
from dmrg.nrg_iter import leftblock_to_next
from dmrg.dmrg_right_to_left import rightblockextend_to_next
from dmrg.dmrg_left_to_right import leftblockextend_to_next

def standard_dmrg(
        model: BaseModel,
        spin_sector: Tuple[int, int], 
        nrg_max_keep: int,
        dmrg_max_keep: List[int]
    ):
    """一个标准的流程\n
    spin_sector: 指定对角化的自旋粒子数，先上后下\n
    nrg_max_keep: 在nrg进行warm up的过程中，最多保留多少个基\n
    dmrg_max_keep: 在dmrg sweep的过程中，最多保留多少个基\n
    dmrg_max_keep应该一共包含model.size - 4个数值，\n
    从1->2,2->3,...,model.size-4->model.size-3。\n
    \n
    在每次更新一个格子的时候，都会把新的ext基上面的算符保留下来，\n
    leftext[model.size-3]和rightext[model.size]就可以构成一个superblock了\n
    每次推进一个格子的时候，新的ext基上的算符会保留，用来做下一次的计算\n
    TODO: DMRG过程中的裁减率以后应该实现，按照数量裁减误差不容易控制
    """
    #这个init_first_site创建DMRGConfig，同时把第一个格子和最后一个
    #格子的哈密顿量和算符存储到DMRGConfig中
    dconf = init_first_site(model, nrg_max_keep)
    #
    #开始warm up, nrg_iter是从一个leftblock到下一个leftblock
    #这个过程中，把leftblockext生成成功并且保存到DMRGConfig
    #因为需要modelsize-3位置上的ext，所以要一直更新到modelsize-2的block
    for phi_idx in range(2, model.size-1):
        #这个phi_idx是目标block的编号
        #推进的时候会新加一个site，这个site就在phi_idx上，
        #找到这个site和已经有的site之间的bond
        newbonds = [bnd for bnd in model.get_site_bonds(phi_idx) if bnd < phi_idx]
        #以后需要用到的算符这次也要存下来
        #这里说需要用到，主要是有两个
        #一是为了nrg算法继续向前推进，构成leftext基上面的哈密顿量时需要用到的算符
        #这一部分是在block基上面的算符，在nrg推进的时候会扩展到ext基上，这一部分会
        #存储在dconf中的left_tmp里面
        #二是以后dmrg计算需要用到的，构成superblock上面的哈密顿量时需要用到的算符
        #这一部分时ext基上的算符
        #这个过程中将block升级到phi_idx，将ext[phi_idx-1]上面的算符保存
        #对于nrg即使phi_idx+1暂时还用不到，phi_idx+m没准要用，也必须要先放到left_tmp中
        #所以这两个需要用到的算符是一致的，都是和大于phi_idx的格子有bond的格子
        site_need_tmp = []
        for stidx in range(1, phi_idx+1):
            for bnd in model.get_site_bonds(stidx):
                if bnd > phi_idx:
                    site_need_tmp.append(stidx)
                    break
        #推进到下一个leftblock
        leftblock_to_next(
            dconf,
            phi_idx,
            newbonds,
            site_need_tmp,
            site_need_tmp
        )
    #在warm up结束了以后，model size-3的leftext就有了，但是这个时候
    #还没有modesize上的rightext，先把这个位置的ext算出来
    extsite = model.size - 1#新加的site
    #这个newbonds是N-1这个site和N这个site的bond
    newbonds = [bond for bond in\
        model.get_site_bonds(extsite) if bond > extsite]
    #以后需要用的算符，这个里面主要是和leftext之间会有链接的
    #在和leftext构成superblock的时候，需要这些算符来构成两个ext之间的hopping
    site_need_tmp = []
    for stidx in range(extsite, model.size+1):
        for bnd in model.get_site_bonds(stidx):
            if bnd < extsite:
                site_need_tmp.append(stidx)
                break
    #准备第一个rightblockextend，这个过程把N上面的rightext保存下来
    _ = prepare_rightblockextend(dconf, model.size, newbonds, site_need_tmp)
    #现在可以开始sweep了
    for _ in range(2):#计算两个sweep
        #开始right sweep
        #这个right sweep的时候，现在只有N上面的ext,从N-1的位置开始
        #推进ext，一直到4上的ext（4上的ext包含到第3个格子，range不包含最后一个）
        for iter_idx, phi_idx in enumerate(range(model.size-1, 3, -1), 0):
            #找到leftblockextend和rightblockextend之间存在的bond
            #现在组成的superblock是phi^phi_idx-2,s^phi_idx-1,s^phi_idx,phi^phi_idx+1
            extrabonds = []
            leftext_edge = phi_idx-1
            for idx in range(1, leftext_edge + 1):
                _bonds = model.get_site_bonds(idx)
                for bnd in _bonds:
                    if bnd > leftext_edge:
                        extrabonds.append((idx, bnd))
            #这次会把rightblockextend[phi_idx+1]升级成rightblock[phi_idx]
            #然后再把rightblock[phi_idx]扩展成rightblockextend[phi_idx],
            #这次扩展会把site[phi_idx-1]添加到rightblock[phi_idx]，这里找到
            #site[phi_idx-1]包含的bond
            newbonds = [bnd for bnd in\
                model.get_site_bonds(phi_idx-1) if bnd > phi_idx-1]
            #rightblockextend[phi_idx]需要包含一些以后用的到的算符
            site_need_tmp = []
            for stidx in range(phi_idx-1, model.size+1):#新加的格子是phi_idx-1
                for bnd in model.get_site_bonds(stidx):
                    if bnd < phi_idx-1:
                        site_need_tmp.append(stidx)
                        break
            #从phi_idx+1推进到phi_idx
            gerg = rightblockextend_to_next(
                dconf, phi_idx, extrabonds, newbonds,
                site_need_tmp, spin_sector, dmrg_max_keep[iter_idx]
            )
            print(gerg)
        #
        #开始left sweep
        #这次从2到N-3
        for iter_idx, phi_idx in enumerate(range(2, model.size-2), 0):
            #找到leftblockextend和rightblockextend之间存在的bond
            #现在组成的superblock是phi^phi_idx-1,s^phi_idx,s^phi_idx+1,phi^phi_idx+2
            extrabonds = []
            leftext_edge = phi_idx
            for idx in range(1, leftext_edge+1):
                _bonds = model.get_site_bonds(idx)
                for bnd in _bonds:
                    if bnd > leftext_edge:
                        extrabonds.append((idx, bnd))
            #这个时候会从leftblockextend[phi_idx-1]升级到leftblock[phi_idx]
            #然后再扩展到leftblockextend[phi_idx]，这时会增加phi_idx+1位置上的site
            #找到这个时候新加到leftblockextend上的bond
            newbonds = [bnd for bnd in\
                model.get_site_bonds(phi_idx+1) if bnd < phi_idx+1]
            #以后会用到的算符
            site_need_tmp = []
            for stidx in range(1, phi_idx+2):
                for bnd in model.get_site_bonds(stidx):
                    if bnd > phi_idx+1:
                        site_need_tmp.append(stidx)
                        break
            #
            gerg = leftblockextend_to_next(
                dconf, phi_idx, extrabonds, newbonds,
                site_need_tmp, spin_sector, dmrg_max_keep[iter_idx]
            )
            print(gerg)

