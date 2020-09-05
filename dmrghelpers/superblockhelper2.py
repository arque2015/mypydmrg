"""使用ProdOp的superblockhelper"""

from multiprocessing import Pool
import numpy
import scipy.linalg
import scipy.sparse
from fermionic.prodop import SuperOp, SuperVec
from fermionic.baseop import Hamiltonian, BaseOperator
from fermionic.superblock import SuperBlockExtend


def merge_left_and_right_ham(leftham: Hamiltonian, rightham: Hamiltonian):
    '''将两个hamiltonian合并成一个ProdOp'''
    leftext = leftham.basis
    rightext = rightham.basis
    superblock = SuperBlockExtend(leftext, rightext)
    superham = SuperOp(superblock)
    superham.add_op_pair(leftham.mat, scipy.sparse.eye(rightext.dim), False)
    superham.add_op_pair(scipy.sparse.eye(leftext.dim), rightham.mat, False)
    return superham


def superham_add_hopping_term(
        superham: SuperOp,
        op1: BaseOperator, op2: BaseOperator,
        coef_t
    ):
    '''增加一个hopping项'''
    #首先得到rightext上面算符产生的额外的符号
    leftext = superham.basis.leftblockextend
    diavals = []
    for col in leftext.iter_idx():
        _pnum = leftext.spin_nums[col]
        _parti_num = numpy.sum(_pnum)
        diavals.append(1.0 if _parti_num % 2 == 0 else -1.0)
    fsign = scipy.sparse.dia_matrix((diavals, 0), op1.mat.shape)
    mat1 = op1.mat * fsign
    mat1 = mat1.multiply(-coef_t)
    mat2 = op2.mat.transpose()
    superham.add_op_pair(mat1, mat2, True)


def __apply_to_row(arg):
    '''作用在某一行上面的结果'''
    superop, idx, supervec = arg
    rowr = numpy.floor_divide(idx, superop.basis.leftblockextend.dim)
    rowl = idx - rowr * superop.basis.leftblockextend.dim
    result = 0.
    for colidx, idxpair in enumerate(supervec.superidxpairs, 0):
        coll = idxpair[0]
        colr = idxpair[1]
        result += superop.extele(rowl, coll, rowr, colr) * supervec.array[colidx]
    return result

def superop_apply_to_supervec(superop: SuperOp, supervec: SuperVec):
    '''将算符作用到vec上'''
    #ret = numpy.ndarray(supervec.array.shape)
    #for retidx, superidx in enumerate(supervec.superidxs, 0):
    #    ret[retidx] = __apply_to_row((superop, superidx, supervec))
    procs = Pool()
    def __args():
        for superidx in supervec.superidxs:
            yield superop, superidx, supervec
    ret = procs.map(__apply_to_row, __args())
    ret = numpy.array(ret)
    procs.close()#Pool有概率会出现死锁的问题，暂时不清楚close能不能解决
    return ret


def supervec_dot_supervec(vec1: SuperVec, vec2: SuperVec):
    '''两个supervec的内积，注意supervec中只保存了特定idx上面的数值'''
    return numpy.dot(vec1.array, vec2.array)

def lanczos(superop: SuperOp, superidxs, step=5, initv='random'):
    '''计算superop的基态\n
    step代表每隔多少步进行一次三对角矩阵的对角化，对角化太频繁了效率低\n
    initv代表初始化的q1向量的数值，如果用random那么就是一个随机的向量\n
    '''
    #
    initvec = SuperVec(superop.basis, superidxs, initv)
    #
    alphas = []
    betas = []
    uvec_arr = None
    qvec = [initvec]
    #想得到一个矩阵的基态时，若找到一个矩阵变换Q = [q1, q2,..., qN]
    #使 HQ = QT，其中T是一个三对角矩阵，若有V是T的本正向量，本正值是t
    #HQV = QTV = tQV，于是QV就是H的一个本征向量，本正值为t
    #也就是说得到T的基态后，H的基态能量就是T的基态能量，H的基态就是QV
    #
    #使用lanczos法得到变换Q
    #首先从初始的q1开始
    preveigval = 250
    while True:
        uvec_arr = superop_apply_to_supervec(superop, qvec[-1])
        if len(qvec) > 1:#在最开始没有beta的数值，这个不用计算
            uvec_arr -= betas[-1] * qvec[-2].array
        alphas.append(numpy.dot(qvec[-1].array, uvec_arr))
        rvec_arr = uvec_arr - alphas[-1] * qvec[-1].array
        betas.append(numpy.sqrt(numpy.dot(rvec_arr, rvec_arr)))
        #向qvec添加一个q，最后会有一个q是多余的，可以优化掉
        qvec.append(SuperVec(superop.basis, superidxs, rvec_arr / betas[-1]))
        #
        if len(alphas) % step != 0:
            continue
        #如果只有一个alpha（在step=1的时候会出现这种可能性)，直接跳过
        if len(alphas) < 2:
            continue
        #对角化T矩阵
        eigval, eigvec =\
            scipy.linalg.eigh_tridiagonal(
                alphas, betas[:-1],
                select='i', select_range=(0, 0)
            )
        #只需要基态
        eigval = eigval[0]
        eigvec = eigvec[:, 0]
        #如果本正值收敛了，就退出循环
        if numpy.abs(eigval - preveigval) < 1e-6:
            break
        preveigval = eigval
    #现在有了V，计算QV
    qvvec = numpy.zeros(len(superidxs))
    for qidx in range(len(alphas)):
        qvvec += eigvec[qidx] * qvec[qidx].array
    return eigval, qvvec
