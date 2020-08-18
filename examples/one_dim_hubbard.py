"""计算一维的Hubbard链"""

from lattice.one_dim_chain import HubbardChain
from dmrg import standard_dmrg


def main():
    '''开始算法'''
    #首先设置格子
    #PBC时，6个格子基态能量-8.0，8个格子-9.65685425
    #在U=1时能量时-6.60115829，U=2时能量-5.40945685
    #在U=3时能量时-4.43335361，U=4时能量-3.66870618
    modelsize = 6
    hubbard = HubbardChain(modelsize, 0.0)
    spin_sector = (modelsize // 2, modelsize // 2)
    #DMRG时的maxkeep
    dkeep = [15 + modelsize*idx for idx in range(modelsize-4)]
    #开始dmrg
    standard_dmrg(
        hubbard, spin_sector, 15, dkeep
    )


if __name__ == "__main__":
    main()
