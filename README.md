dmrg算法演示
======

按照computational many particle physics书中第21章实现

TODO:
------

+ ~~实现matrix product，superblock。实现基还有其上的算符表示~~

+ ~~实现NRG算法，用来做warm up~~

+ 实现DMRG算法，计算一个没有U的一维Hubbard链  

### 可以考虑的优化

避免过早优化，在上述三个功能完成前不实现优化

+ rightext_hamiltonian_to_superblock和rightext_oper_to_superblock还是非常慢，考虑多线程  
以及算法的优化

+ 算符是十分稀疏的，使用完整的矩阵保存浪费了很多内存，实现或调用一些稀疏矩阵的工具

+ DMRGConfig中的-ext_storage使用频率不高，在可以存到硬盘里，节约内存的使用

### 备忘

+ [left/right-block的递推](markdowns/left_right_block.md#block的递推)

+ [调用结构](markdowns/program_struct.md#调用结构)

+ [DMRGConfig](markdowns/program_struct.md#DMRGConfig)
