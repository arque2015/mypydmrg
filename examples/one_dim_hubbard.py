"""计算一维的Hubbard链"""

import tracemalloc
from lattice.one_dim_chain import HubbardChain
from dmrg import standard_dmrg


def main():
    '''开始算法'''
    tracemalloc.start()
    #首先设置格子
    #PBC时，6个格子基态能量-8.0，8个格子-9.65685425
    #在U=1时能量时-6.60115829，U=2时能量-5.40945685
    #在U=3时能量时-4.43335361，U=4时能量-3.66870618
    #关联函数6个格子：
    #U=0: Sz_i*Sz_i = 0.5, Sz_i*Sz_(i+-1) = -0.2222 Sz_i*Sz_(i+-3) = -0.0555
    #U=1: Sz_i*Sz_i = 0.5677 Sz_i*Sz_(i+-1) = -0.2669 Sz_i*Sz_(i+-2) = 0.0217
    #Sz_i*Sz_(i+-3) = -0.07723
    #U=2: Sz_i*Sz_i = 0.6384 Sz_i*Sz_(i+-1) = -0.3199 Sz_i*Sz_(i+-2) = 0.0549
    #Sz_i*Sz_(i+-3) = -0.1085
    #U=3: Sz_i*Sz_i = 0.7107 Sz_i*Sz_(i+-1) = -0.3787 Sz_i*Sz_(i+-2) = 0.0972
    #Sz_i*Sz_(i+-3) = -0.1478
    #U=4: Sz_i*Sz_i = 0.7779 Sz_i*Sz_(i+-1) = -0.4354 Sz_i*Sz_(i+-2) = 0.1403
    #Sz_i*Sz_(i+-3) = -0.1876
    modelsize = 6
    hubbard = HubbardChain(modelsize, 4.0)
    spin_sector = (modelsize // 2, modelsize // 2)
    measures = [('sz', 1), ('sz', 2), ('sz', 3)]#, ('sz', 4), ('sz', 5), ('sz', 6)]
    #DMRG时的maxkeep
    dkeep = [15 + modelsize*idx for idx in range(modelsize-4)]
    #开始dmrg
    standard_dmrg(
        hubbard, spin_sector, 15, dkeep, measures
    )
    #内存使用状况
    print('峰值内存 ', tracemalloc.get_traced_memory()[1])
    snapshot = tracemalloc.take_snapshot()
    sta = snapshot.statistics('filename')
    for msta in sta[:10]:
        print(msta)
    #print(sta[:10])



if __name__ == "__main__":
    main()
