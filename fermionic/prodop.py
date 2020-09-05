"""
两个基组合而成的一个算符
"""

import traceback
import numpy
import scipy.sparse
from basics.operator import Operator
from .superblock import SuperBlockExtend

class SuperOp(Operator):
    """<i,k| P Q |j,l> = <i|P|j> <k|Q|l>\n
    这个算符包含一组这样的算符求和:\n
    H = sum_k P_k Q_k\n
    注意如果有反对易符号的要求，应该在P算符中处理好\n
    """
    def __init__(self, basis: SuperBlockExtend):
        super().__init__()
        self._basis = basis
        self._pair_count = 0
        self._plist = []
        self._qlist = []
        self._trans = []
        self._pnonzs = []
        self._qnonzs = []
        self._ldim = basis.leftblockextend.dim * basis.rightblockextend.dim
        self._rdim = self._ldim

    @property
    def basis(self):
        '''所在的基'''
        return self._basis

    @property
    def mat(self):
        '''获得矩阵'''
        #mat会将矩阵全部计算出来，效率比较差，而且这时应该利用
        #superblockhelper中的方法，利用Hamiltonian中的成员函数
        #没有必要使用superblockhelper2和SuperOp提供的方法
        tst = traceback.extract_stack()
        print('%s\n计算的时候不应该调用mat属性' % str(tst[-2]))
        #
        ret = scipy.sparse.lil_matrix((self._ldim, self._rdim))
        #
        for lidx in range(self._ldim):
            for ridx in range(self._rdim):
                val = self.ele(lidx, ridx)
                if val != 0:
                    ret[lidx, ridx] = val
        return ret

    def add_op_pair(
            self,
            pop: scipy.sparse.spmatrix,
            qop: scipy.sparse.spmatrix,
            contain_trans
        ):
        '''增加一对算符\n
        pop是在左边的基上面的，qop是在右边的基上面的，\n
        contain_trans代表是否要增加这组算符的转置到superop上面\n
        这个对增加hopping项比较方便\n\n
        ``````
        增加hopping的时候需要处理好右边的算符增加的反对易的符号\n
        superblockhelper2中有superham_add_hopping_term方法\n
        是处理好的
        '''
        if scipy.sparse.isspmatrix_coo(pop):
            pop = pop.tolil()
        if scipy.sparse.isspmatrix_coo(qop):
            qop = qop.tolil()
        self._plist.append(pop)
        self._qlist.append(qop)
        self._trans.append(contain_trans)
        pl1, pl2 = pop.nonzero()
        self._pnonzs.append(
            [lef + rig * self._basis.leftblockextend.dim\
                for lef, rig in zip(pl1, pl2)]
        )
        ql1, ql2 = qop.nonzero()
        self._qnonzs.append(
            [lef + rig * self._basis.rightblockextend.dim\
                for lef, rig in zip(ql1, ql2)]
        )
        self._pair_count += 1


    def ele(self, lidx, ridx):
        '''获得一个元素'''
        leftdim = self._basis.leftblockextend.dim
        qlidx = numpy.floor_divide(lidx, leftdim)
        plidx = lidx - qlidx * leftdim
        qridx = numpy.floor_divide(ridx, leftdim)
        pridx = ridx - qridx * leftdim
        return  self.extele(plidx, pridx, qlidx, qridx)


    def extele(self, il1, il2, ir1, ir2):
        '''根据leftext和rightext上的指标来计算数值'''
        result = 0.
        for pairidx in range(self._pair_count):
            trans = self._trans[pairidx]
            pnz = self._pnonzs[pairidx]
            qnz = self._qnonzs[pairidx]
            if not self.check_extidx_valid(pnz, il1, il2, qnz, ir1, ir2, trans):
                continue
            p_op = self._plist[pairidx]
            q_op = self._qlist[pairidx]
            if scipy.sparse.isspmatrix_dia(p_op):#如果左边的算符是单位算符
                if il1 == il2:
                    result += q_op[ir1, ir2] * p_op.diagonal(0)[il1]
                    if trans:
                        result += q_op[ir2, ir1] * p_op.diagonal(0)[il1]
                continue
            if scipy.sparse.isspmatrix_dia(q_op):#如果右边的算符是单位算符
                if ir1 == ir2:
                    result += p_op[il1, il2] * q_op.diagonal(0)[ir1]
                    if trans:
                        result += p_op[il2, il1] * q_op.diagonal(0)[ir1]
                continue
            result += p_op[il1, il2] * q_op[ir1, ir2]
            if trans:
                result += p_op[il2, il1] * q_op[ir2, ir1]
        return result


    def check_extidx_valid(self, pnz, il1, il2, qnz, ir1, ir2, trans):
        '''检验是否两个值都不是0'''
        hasval = False
        pidx = il1 + il2 * self._basis.leftblockextend.dim
        qidx = ir1 + ir2 * self._basis.rightblockextend.dim
        hasval = pidx in pnz and qidx in qnz
        #如果要把transpose也加上的话，查看transpose是不是有数值
        if not hasval and trans:
            pidx = il2 + il1 * self._basis.leftblockextend.dim
            qidx = ir2 + ir1 * self._basis.rightblockextend.dim
            hasval = pidx in pnz and qidx in qnz
        return hasval

    def __str__(self):
        template = 'SuperOp: \n'
        for p_op, q_op in zip(self._plist, self._qlist):
            template += p_op.__class__.__name__ + '\t'
            template += q_op.__class__.__name__ + '\n'
        return template

class SuperVec():
    """用来表示一个|i,j>这样的向量\n
    因为在对角化的时候是要分块的(指定spin sector)，\n
    这时候会有一个superidxs,这个类中会保存这个superidxs，\n
    用在以后的phival中\n
    """
    def __init__(self, basis, superidxs, val):
        self._basis = basis
        self._superidxs = superidxs
        if isinstance(val, numpy.ndarray):
            self._val = val
        if isinstance(val, str):
            if val == 'random':
                self._val = numpy.random.random([len(superidxs)])
                self._val /= numpy.sqrt(numpy.dot(self._val, self._val))
        #
        self._superidxpairs = []
        for sidx in superidxs:
            rightidx = numpy.floor_divide(sidx, basis.leftblockextend.dim)
            leftidx = sidx - rightidx * basis.leftblockextend.dim
            self._superidxpairs.append((leftidx, rightidx))
        #

    @property
    def array(self):
        '''返回数组'''
        return self._val

    @property
    def superidxs(self):
        '''这组向量对应的superblock上的idx'''
        return self._superidxs

    @property
    def superidxpairs(self):
        '''返回superidxs对应的leftext上的编号和rightext上的编号'''
        return self._superidxpairs

    def __str__(self):
        template = 'SuperVec: \n'
        template += str(self._superidxs)
        return template
