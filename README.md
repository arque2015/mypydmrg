dmrg算法演示
======

按照computational many particle physics书中第21章实现

MPS和MPO按照The density-matrix renormalization group in the age of
matrix product states 实现

TODO:
------ 

+ 实现MPS的表示，并且实现用MPS表示一个Block基

+ 实现用MPO的表示，表示升降算符，验证MPO作用到MPS上的效果

传统DMRG
------

+ 在做DMRG的时候，优先以密度矩阵的求和为依据而不是maxkeep，只依据数量误差基本不能控制   

+ warm up改称无限格子时的DMRG算法

### 可以考虑的优化

+ 算符是十分稀疏的，使用完整的矩阵保存浪费了很多内存，实现或调用一些稀疏矩阵的工具

+ DMRGConfig中的-ext_storage使用频率不高，在可以存到硬盘里，节约内存的使用

+ 给各个部分加上计时

### 备忘

+ [left/right-block的递推](https://github.com/maryprimary/mypydmrg/wiki/left_right_block#block的递推)

+ [调用结构](https://github.com/maryprimary/mypydmrg/wiki/program_struct#调用结构)

+ [DMRGConfig](https://github.com/maryprimary/mypydmrg/wiki/program_struct#DMRGConfig)

+ [DMRG迭代](https://github.com/maryprimary/mypydmrg/wiki/dmrg_sweep)
