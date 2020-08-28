"""计算一维的ladder"""

from lattice.one_dim_ladder import HubbardLadder
from dmrg import standard_dmrg


def main():
    '''开始计算'''
    #晶格的配置
    lenx = 2
    modelsize = 3 * lenx
    spin_sector = (modelsize // 2, modelsize // 2)
    hubbard = HubbardLadder(lenx, 4.0, 0.05)
    measures = [('nu', 1), ('nu', 2), ('nu', 3), ('nd', 4), ('nd', 5), ('nd', 6)]
    #ED算基态能量的结果
    #U=0时结果是-12.63841536，U=1时是-11.13668625
    #U=2时结果是-9.78120910，U=4时是-7.52100677
    #关联函数
    #U=0: Sz_1*Sz_1 = 0.4876 Sz_1*Sz_2 = -0.2172 Sz_1*Sz_3 = -0.0124
    #Sz_1*Sz_4 = -0.1707 Sz_1*Sz_5 = -0.0008 Sz_1*Sz_6 = -0.0864
    #U=1: Sz_1*Sz_1 = 0.5395 Sz_1*Sz_2 = -0.2463 Sz_1*Sz_3 = 0.0044
    #Sz_1*Sz_4 = -0.2012 Sz_1*Sz_5 = 0.0095 Sz_1*Sz_6 = -0.1060
    #U=2: Sz_1*Sz_1 = 0.5914 Sz_1*Sz_2 = -0.2797 Sz_1*Sz_3 = 0.0276
    #Sz_1*Sz_4 = -0.2353 Sz_1*Sz_5 = 0.0245 Sz_1*Sz_6 = -0.1285
    #U=4: Sz_1*Sz_1 = 0.6953 Sz_1*Sz_2 = -0.3570 Sz_1*Sz_3 = 0.0901
    #Sz_1*Sz_4 = -0.3184 Sz_1*Sz_5 = 0.0671 Sz_1*Sz_6 = -0.1771
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
        hubbard, spin_sector, 15, dkeep, measures
    )


if __name__ == "__main__":
    main()
