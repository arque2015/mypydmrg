程序设计时的结构
======

## 调用结构

1. lattice和basics两个文件夹
>  basics中是关于基，算符，最基本的一些工具。lattice是和晶格有关的工具，  
>  lattice和算法本身没有关联。他们除了本身文件夹下的内容不应该调用其他文  
>  件夹下的代码。

2. fermionic文件夹
>  这个文件夹下面是DMRG算法中常用的基本数据类型，这下面的代码可以引用basics
>  中的代码，和lattice中的内容不应该有联系，因为这和晶格没有关系。

3. dmrghelper文件夹
>  这里面的内容旨在简化fermionic中的代码的调用，封装常用的调用方式，从  
>  结构上说可以调用basics和fermionic中的代码。

4. dmrg文件夹
>  这个是最高层次的调用，可以调用上面3个中所有的代码。包括lattice。


## DMRGConfig

这个类中保存一些计算过程中需要保留一段时间的变量，因为在一个sweep中，需要用到上  
一个sweep的哈密顿量和算符，主要存储的是-BlockExtend基下的哈密顿量和算符，因为  
[block的递推](https://github.com/maryprimary/mypydmrg/wiki/left_right_block#block的递推)这个过程中，superblock是从-BlockExtend建立的，算符  
的递推也是从-BlockExtend基上进行的。在更新-Block的时候，需要保存一个-Block，  
他要用来计算他自己的-BlockExtend，以及这个-BlockExtend上的哈密顿量和算符，还  
要从这个-BlockExtend上计算下一个-Block，这个Block是需要保存并且迭代的。
