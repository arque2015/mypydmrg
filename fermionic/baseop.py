"""
在Left/Right-Block中的算符的基本属性
"""

import numpy
from basics.operator import Operator

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
        mat = numpy.matmul(op1.mat, op2.mat.transpose())
        # + C^+_2 C_1
        mat = mat + mat.transpose()
        # t系数
        mat = -coeft * mat
        self.addnewterm(mat)


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
