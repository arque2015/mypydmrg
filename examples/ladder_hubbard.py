"""计算一维的ladder"""

from lattice.one_dim_ladder import HubbardLadder
from dmrg import standard_dmrg


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
    #DMRG时的maxkeep
    #left sweep的时候phi_idx从2...到modelsize - 3
    #这个phi_idx是ext的编号，所以共有modelsize-4次从ext升级到下一个ext
    #的过程，这个过程中，会从ext升级到block，保存多少个基，由dkeep指定
    dkeep = [20 + modelsize*idx for idx in range(modelsize-4)]
    #开始dmrg
    standard_dmrg(
        hubbard, spin_sector, 15, dkeep
    )


if __name__ == "__main__":
    main()
