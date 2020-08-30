"""
在Left/Right-Block中的算符的基本属性
"""

import numpy
import scipy.sparse
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
        if not scipy.sparse.issparse(val):
            raise ValueError('val不是稀疏矩阵')
        self._mat = val
        shape = numpy.shape(val)
        self._ldim = shape[0]
        self._rdim = shape[1]

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

    def get_block(self, idxs):
        '''获取一个块'''
        #
        ret = self._mat.tocsr()[idxs]
        ret = ret.tocsc()[:, idxs]
        return ret

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
        self._ldim = basis.dim
        self._rdim = basis.dim
        if not scipy.sparse.issparse(mat):
            raise ValueError('mat不是稀疏矩阵')
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
        if not scipy.sparse.issparse(newmat):
            raise ValueError('newmat不是稀疏矩阵')
            #print('newmat不是稀疏矩阵')
            #newmat = scipy.sparse.csr_matrix(newmat)
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
        mat = op1.mat * op2t#umpy.matmul(op1.mat, op2t)
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
        #右乘一个有正负号的单位矩阵上去
        if not isinstance(self._basis, SuperBlockExtend):
            raise ValueError('只能给superblock用')
        leftext = self._basis.leftblockextend
        rightext = self._basis.rightblockextend
        diavals = []
        for col in leftext.iter_idx():
            _pnum = leftext.spin_nums[col]
            _parti_num = numpy.sum(_pnum)
            diavals.append(1.0 if _parti_num % 2 == 0 else -1.0)
        fsign = scipy.sparse.dia_matrix((diavals, 0), op1.mat.shape)
        mat1 = op1.mat * fsign
        mat1 = mat1.multiply(-coeft).tocsr()
        mat2 = op2.mat.tocsr()
        #最后reshape成结果的形状，这个时候是先遍历ld1和ld2的，所以
        #和需要的（ld1*rd1, ld2*rd2）是一样的
        _dim = leftext.dim * rightext.dim
        #先构造mato1
        #mato1 = scipy.sparse.dok_matrix((_dim, _dim))
        #idxilist, idxjlist = mat1.nonzero()
        #idxllist, idxklist = mat2.nonzero()
        #for idxi, idxj in zip(idxilist, idxjlist):
        #    for idxl, idxk in zip(idxllist, idxklist):
        #        mato1[idxk * leftext.dim + idxi, idxl * leftext.dim + idxj]\
        #            = mat1[idxi, idxj] * mat2[idxl, idxk]
        #使用外积的方式构造mato1
        mat1f = mat1.reshape((leftext.dim * leftext.dim, 1))
        mat2f = mat2.transpose().reshape((1, rightext.dim * rightext.dim))
        #
        mato1f = mat1f * mat2f
        #(l1*l2, r1*r2) -> (r1*l1, r2*l2)
        #这个时候每个(l1, l2)的矩阵就是一个小块
        mato1f = mato1f.tocsc()
        block_arr = numpy.array([[None]*rightext.dim]*rightext.dim)
        idxr2list, idxr1list = mat2.nonzero()#因为要做transpose，2，1翻过来
        for idxr1, idxr2 in zip(idxr1list, idxr2list):#range(rightext.dim):
            block_ent = mato1f[:, idxr1*rightext.dim + idxr2]\
                    .reshape((leftext.dim, leftext.dim))
            block_arr[idxr1, idxr2] = block_ent
            block_arr[idxr2, idxr1] = block_ent.transpose()
        for idxr in range(rightext.dim):
            if block_arr[idxr, idxr] is None:
                block_arr[idxr, idxr] = scipy.sparse.dok_matrix((leftext.dim, leftext.dim))
        mato = scipy.sparse.bmat(block_arr)
        #assert numpy.allclose(mato1b.toarray(), mato1.toarray())
        #再构造mato2
        #mato2 = scipy.sparse.dok_matrix((_dim, _dim))
        #idxklist, idxllist = idxllist, idxklist
        #idxjlist, idxilist = idxilist, idxjlist
        #for idxk, idxl in zip(idxklist, idxllist):
        #    for idxj, idxi in zip(idxjlist, idxilist):
        #        mato2[idxk *leftext.dim + idxi, idxl * leftext.dim + idxj]\
        #            = mat2[idxk, idxl] * mat1[idxj, idxi]
        #不使用外积的方式构造mato2，直接使用转置，因为这时的copy快很多
        self.addnewterm(mato)
        return mato

    def add_u_term(self, opu: BaseOperator, coef_u):
        '''添加一个U项'''
        if opu.basis.dim != self.basis.dim:
            raise ValueError('opu的dim对不上')
        self.addnewterm(coef_u * opu.mat)

    def add_mu_term(self, opmu, coef_mu):
        '''添加一个mu项'''
        if opmu.basis.dim != self.basis.dim:
            raise ValueError('opu的dim对不上')
        self.addnewterm(coef_mu * opmu.mat)

    def get_block(self, idxs):
        '''获得哈密顿量中的一个block
        注意这里返回的不是一个Hamiltonian，
        这个时候再创建一个新的Basis没有必要（？），
        调用者自己管理对应的基
        '''
        #
        #mat = scipy.sparse.dok_matrix((len(idxs), len(idxs)))
        #fullmat = None
        #if scipy.sparse.isspmatrix_coo(self._mat):
        #    fullmat = self._mat.todok()
        #else:
        #    fullmat = self._mat
        #for newidx1, idx1 in enumerate(idxs, 0):
        #    for newidx2, idx2 in enumerate(idxs, 0):
        #        mat[newidx1, newidx2] = fullmat[idx1, idx2]
        #mat = mat.tocsr()
        ret = self._mat.tocsr()[idxs]
        ret = ret.tocsc()[:, idxs]
        return ret
