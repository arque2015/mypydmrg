DMRG迭代的过程
======

## 密度矩阵
和书上略有不同的是，这里把基态的波函数变形成一个矩阵，就是说（21.36）中右手边的  
C(alpha,s_j,s_j+1,beta)，改成C(aplha,s_j;s_j+1,beta)这样一个矩阵，这时  
右手边的求和就可以写成C^T x C了。左向右的时候是一样的。

## 迭代
迭代的过程中，通过新的基构造算符这个过程被包含在rightblockextend_to_next和  
leftblockextend_to_next之中了，这个时候的裁减误差是在下一次观测的时候才会  
体现出来的，每次这个方法把一个-blockextend推进到下一个-blockextend之后，并  
没有作出观测，但是这个方法的一开始，需要通过上一次算的-blockextend来得到super-  
block，这个时候做观测正好。
