"""
在Left/Right-Block中的算符的基本属性
"""

import numpy
from basics.operator import Operator
from fermionic.superblock import SuperBlockExtend

class BaseOperator(Operator):
    """一个通用的单个格子的算符
    stidx： 格子编号
    spin： 自旋+1或者-1
    bss： 所在的基
    val: 矩阵
    还有具体的矩阵，如果是在fock space上，可以直接调用create_from_sitebasis"""
    def __init__(self, stidx, bss, isferm, val, spin=0):
        '''这里只保存基础的属性'''
        super().__init__()
        self._stidx = stidx
        self._basis = bss
        self._spin = spin
        self._isferm = isferm
        self._mat = val

    @property
    def mat(self):
        '''整个矩阵'''
        return self._mat

    @property
    def basis(self):
        '''所在的基'''
        return self._basis

    @property
    def siteidx(self):
        '''算符的指标'''
        return self._stidx

    @property
    def spin(self):
        '''自旋的指标'''
        return self._spin

    @property
    def isferm(self):
        '''是不是反对易'''
        return self._isferm

    def ele(self, lidx, ridx):
        '''返回某个矩阵元'''
        return self._mat[lidx, ridx]

    def __str__(self):
        template = "BaseOperator: \n"
        template += 'Basis: %s\n' % self._basis.__class__.__name__
        template += 'Site: %d Spin: %s\n' % (self._stidx, self._spin)
        template += 'mat:\n'
        template += str(self._mat)
        return template


class Hamiltonian(Operator):
    """哈密顿量的内容
    """
    def __init__(
            self,
            basis,
            mat
        ):
        super().__init__()
        self._basis = basis
        self._mat = mat

    @property
    def basis(self):
        '''所在的基'''
        return self._basis

    @property
    def mat(self):
        '''算符的矩阵'''
        return self._mat

    def addnewterm(self, newmat):
        '''给现在的矩阵增加新的内容'''
        self._mat += newmat

    def ele(self, lidx, ridx):
        '''矩阵元'''
        return self._mat[lidx, ridx]

    def __str__(self):
        template = 'Hamiltonian: \n'
        template += 'Basis: %s\n' % self._basis.__class__.__name__
        template += str(self._mat)
        return template

    #@profile#使用line_profiler进行性能分析的时候把这个注释去掉
    def add_hopping_term(
            self,
            op1: BaseOperator, op2: BaseOperator,
            coeft
        ):
        '''增加一个hopping项
        op1和op2应当是两个格子的产生算符，coeft是这个bond的强度
        ``````
        Issue#11: 增加bond大小的设置
        '''
        if op1.basis.dim != self.basis.dim:
            raise ValueError('op1的dim对不上')
        if op2.basis.dim != self.basis.dim:
            raise ValueError('op2的dim对不上')
        # C^+_1 C_2
        op2t = op2.mat.transpose()
        mat = numpy.matmul(op1.mat, op2t)
        # + C^+_2 C_1
        matt = mat.transpose()
        mat = mat + matt
        # t系数
        mat = -coeft * mat
        self.addnewterm(mat)
        return mat


    #@profile
    def superblock_add_hopping_term(
            self,
            op1: BaseOperator, op2: BaseOperator,
            coeft
        ):
        '''在superblock上添加hopping项目\n
        op1必须是leftblockext上面的算符，op2也必须是rightblockext\n
        Issue#16：优化速度
        '''
        #先将left算符整理成平的，在整理之前先把列的粒子数统计
        #这个和反对易的符号有关系
        if not isinstance(self._basis, SuperBlockExtend):
            raise ValueError('只能给superblock用')
        leftext = self._basis.leftblockextend
        rightext = self._basis.rightblockextend
        mat1 = numpy.ndarray(numpy.shape(op1.mat))
        for col in leftext.iter_idx():
            _pnum = leftext.spin_nums[col]
            _parti_num = numpy.sum(_pnum)
            if _parti_num % 2 == 0:
                mat1[:, col] = op1.mat[:, col]
            else:
                mat1[:, col] = -op1.mat[:, col]
        #将左右两个算符整理成向量
        #mat1 = numpy.reshape(mat1, [numpy.square(leftext.dim)])
        #mat2 = numpy.reshape(op2.mat.transpose(), [numpy.square(rightext.dim)])
        ##给两个向量做外积，这个时候出来的矩阵的形状是（ld1*ld2, rd1*rd2）
        ##目标的形状是（ld1*rd1, ld2*rd2)
        #mato = numpy.outer(mat1, mat2)
        ##给现在的reshape,把ld1,ld2,rd1,rd2分开
        #mato = numpy.reshape(mato,\
        #    [leftext.dim, leftext.dim,\
        #        rightext.dim, rightext.dim])
        ##调整顺序，注意在numpy中存储的时候，是先遍历靠后的指标的，
        ##所以调整成（rd1, ld1, rd2, ld2)
        #mato = numpy.transpose(mato, [2, 0, 3, 1])
        #用einsum不会让reshape变的特别慢，虽然reshape还是很慢
        #为什么einsum快不知道
        #mat2 = op2.mat.transpose()#einsum中改顺序了，这个时候用哪个都差不多速度区别不大
        mat2 = op2.mat
        mato1 = numpy.einsum('ij,lk->kilj', mat1, mat2)
        #最后reshape成结果的形状，这个时候是先遍历ld1和ld2的，所以
        #和需要的（ld1*rd1, ld2*rd2）是一样的
        _dim = leftext.dim * rightext.dim
        #mato1 = numpy.reshape(mato1, [_dim, _dim])
        #加上他的复共厄（纯实所以是转置）
        #mato = numpy.random.randn(_dim, _dim)
        #matot = numpy.random.randn(_dim, _dim)
        #matot = mato.transpose()#numpy.random.randn(_dim, _dim)
        #matot = mato[::-1]
        #numpy.copy(mato).transpose()#mato.T#transpose()
        #mato = mato + matot#numpy.add(mato, matot)#
        #for idx in range(_dim):
        #    mato[idx, idx:] = mato[idx, idx:] + mato[idx:, idx]
        #    mato[idx:, idx] = mato[idx, idx:]
        #mato = add_transpose_to(mato)
        #利用转置会触发copy，非常慢
        mato2 = numpy.einsum('kl,ji->kilj', mat2, mat1)
        #mato2 = numpy.reshape(mato2, [_dim, _dim])
        mato = mato1 + mato2
        mato = numpy.reshape(mato, [_dim, _dim])
        # t系数
        mato = -coeft * mato
        self.addnewterm(mato)
        return mato

    def add_u_term(self, opu: BaseOperator, coef_u):
        '''添加一个U项'''
        if opu.basis.dim != self.basis.dim:
            raise ValueError('opu的dim对不上')
        self.addnewterm(coef_u * opu.mat)


    def get_block(self, idxs):
        '''获得哈密顿量中的一个block
        注意这里返回的不是一个Hamiltonian，
        这个时候再创建一个新的Basis没有必要（？），
        调用者自己管理对应的基
        '''
        mat = numpy.zeros([len(idxs), len(idxs)])
        for newidx1, idx1 in enumerate(idxs, 0):
            for newidx2, idx2 in enumerate(idxs, 0):
                mat[newidx1, newidx2] = self._mat[idx1, idx2]
        return mat
