dmrg算法演示
======

按照computational many particle physics书中第21章实现

## 说明

这个程序主要是演示了DMRG算法的思路，在 `expmples` 文件夹中有一些例子  
直接运行即可
```bash
    python3 examples/one_dim_hubbard.py
```

### 晶格设置

若要实现一个其他的晶格，可以在 `lattice` 文件夹中新建一个文件，比如 `custom.py`  
创建一个继承 `lattice.BaseModel` 的类并至少重写他的 `__init__` 方法  


```python
"""custom.py"""

from . import BaseModel

class Custom(BaseModel):
    """定义晶格的类，这里举例一个6个格子的链"""
    def __init__(self):
        #这里需要所有的格子的编号，一般就是1,2,3...,N，这里就是1，2，3，4,5,6
        super().__init__([1, 2, 3, 4, 5, 6])
        #然后把所有的bond放到self._bonds之中
        self._bonds[1] = [2, 6]
        self._bonds[2] = [3, 1]
        self._bonds[3] = [4, 2]
        self._bonds[4] = [5, 3]
        self._bonds[5] = [6, 4]
        self._bonds[6] = [1, 5]
        #这里不需要考虑bond是不是定义重复了，在构造哈密顿量的时候，
        #leftblock只从大（新加的site）到小（block中已经有的site)
        #rightblock只从小到大，superblock只从左到右

    def get_t_coef(self, st1, st2):
        '''重写这个方法来设置bond之间的大小，st1和st2是两个格子的编号\n
        注意这个是会增加负号的，不需要加负号
        '''
        return 1.0
```

在设置完晶格以后，就可以进行DMRG的计算了，可以在 `examples` 中创建一个 `custom_hubbard.py`  

```python
"""custom_hubbard.py"""

from lattice.custom import Custom
from dmrg import standard_dmrg

def main():
    '''计算lattice/custom.py中定义的模型'''
    hubbard = Custom()
    spin_sector = (3, 3)
    #leftext: 1->2, 2->3.rightext: 6->5, 5->4.一共裁减了两次，设置两个数值
    dkeep = [15, 20]
    #nrg做warm up的时候最多保留多少个基
    nkeep = 15
    #开始dmrg
    standard_dmrg(
        hubbard, spin_sector, nkeep, dkeep
    )

if __name__ == "__main__":
    main()

```

在完成这两个步骤以后  
```bash
    python3 examples/custom_hubbard.py 
```
就可以进行计算了  

在计算更大，更复杂的格子的时候，应该改进两个`python`文件里的代码来实现想要的功能。  
  


### 运行要求

程序完全使用python3实现，此外需要 `numpy` 库，安装好python3和numpy即可。
  


### 备忘

+ [left/right-block的递推](https://github.com/maryprimary/mypydmrg/wiki/left_right_block#block的递推)

+ [调用结构](https://github.com/maryprimary/mypydmrg/wiki/program_struct#调用结构)

+ [DMRGConfig](https://github.com/maryprimary/mypydmrg/wiki/program_struct#DMRGConfig)

+ [DMRG迭代](https://github.com/maryprimary/mypydmrg/wiki/dmrg_sweep)

+ [迭代是基的升级](https://github.com/maryprimary/mypydmrg/wiki/get_phival)
