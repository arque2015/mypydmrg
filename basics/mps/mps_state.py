"""表示一个MPS"""

import numpy
from . import MTensor, MPSContainer


class MPSState(MPSContainer):
    '''表示一个MPS态'''
    def __init__(
            self,
            leftinit: MTensor,
            rightinit: MTensor,
            total_len
        ):
        #lshape = numpy.shape(leftinit)
        #print(lshape)
        #ltensor = MTensor(lshape[0], 1, lshape[1], 1, leftinit)
        self._tot_len = total_len
        super().__init__(leftinit)
        #rtensor = MTensor(lshape[0], lshape[1], 1, total_len, rightinit)
        self.add_site(rightinit)


    def left_shrink(self, sitecode):
        '''获得在sitecode对应的分量上面的数值'''
        ret = super().left_shrink(sitecode)
        if numpy.prod(numpy.shape(ret)) != 1:
            raise RuntimeError('没有收缩到最后')
        return ret[0, 0]


def create_left_canonical(length, entity):
    '''将一个态转换成MPS表示'''
    #tot_dim = numpy.prod(numpy.shape(entity))
    a_l_1 = 1
    a_l = None
    #
    if not isinstance(entity, numpy.ndarray):
        raise ValueError('entity不是ndarray')
    phi_mat = entity
    mps_dict = {}
    for idx in range(1, length+1):
        #左边的指标现在是（a_l-1, delta_l)
        leftdim = a_l_1 * 4
        #现在右边的指标就是(delta_l+1,...,delta_L)
        rightdim = numpy.power(4, length-idx)
        #把现在的phi重新整理形状
        #现在的形状是（a_l-1),(delta_l,...delta_L)
        #reshape的结果就是按照低位高位进行排列的
        phi_mat = numpy.reshape(phi_mat, [leftdim, rightdim])
        #print(phi_mat.shape)
        #al是分解的时候较小的那一个
        a_l = numpy.minimum(leftdim, rightdim)
        umat, sdia, vmat = numpy.linalg.svd(phi_mat, full_matrices=False)
        smat = numpy.diag(sdia)
        #print(umat, smat, vmat)
        #print(umat.shape, smat.shape, vmat.shape)
        mps_dict[idx] =\
            MTensor(a_l_1, a_l, idx, umat)
        phi_mat = numpy.matmul(smat, vmat)
        a_l_1 = a_l
    #print('norm: ', numpy.matmul(smat, vmat))
    norm = numpy.matmul(smat, vmat)[0][0]
    #print(mps_dict)
    ret = MPSState(mps_dict[1], mps_dict[length], length)
    for idx in range(2, length):
        ret.add_site(mps_dict[idx])
    #print(mps_dict)
    return ret, norm


def create_random_mps(length, maxd):
    '''创建一个随机的mps'''
    a_l_1 = 1
    a_l = None
    #
    mps_dict = {}
    for idx in range(1, length+1):
        #计算这个时候右边的指标
        #这个的指标不大于maxd
        rightdim = numpy.power(4, length-idx)
        a_l = 4 * a_l_1
        a_l = numpy.min([rightdim, a_l, maxd])
        #随机生成一个张量
        ranten = numpy.random.randn(a_l_1, 4, a_l)
        mps_dict[idx] = MTensor(a_l_1, a_l, idx, ranten)
        a_l_1 = a_l
    #
    ret = MPSState(mps_dict[1], mps_dict[length], length)
    for idx in range(2, length):
        ret.add_site(mps_dict[idx])
    return ret


def make_mps_left_canonical(mps: MPSState):
    '''让一个mps变成left canonical的形式'''
    length = mps._tot_len
    mps_dict = {}
    s_l_1 = 1
    #s_l = None
    leftprefix = numpy.eye(1)
    for idx, tensor in mps.left_iter():
        #print(idx, tensor)
        entity = tensor.tensor_entity
        shape = tensor.tensor_shape
        #把上一次的SV^+乘上来，收缩掉a_l-1
        mat = numpy.reshape(entity, [shape[0], shape[1] * shape[2]])
        mat = numpy.matmul(leftprefix, mat)
        #现在的shape是s_l-1,(delta, a_l)，把delta放到左边
        #重新整理形状
        leftdim = s_l_1 * shape[1]
        rightdim = shape[2]
        s_l = numpy.minimum(leftdim, rightdim)
        #
        mat = numpy.reshape(mat, [leftdim, rightdim])
        #svd
        amat, sdia, vmat = numpy.linalg.svd(mat, full_matrices=False)
        #现在A是[（s_l-1,delta）, s_l)],S是[s_l],V是[s_l, a_l]
        #保存新的A矩阵，生成新的SV^+
        aten = MTensor(s_l_1, s_l, idx, amat)
        for row in range(s_l):
            vmat[row, :] *= sdia[row]
        ##验证矩阵
        #unimat = numpy.zeros([s_l, s_l])
        #for ide in range(4):
        #    dmat = aten.delta_of(ide)
        #    unimat += numpy.matmul(dmat.transpose(), dmat)
        ##print(unimat)
        #print(numpy.allclose(numpy.eye(s_l), unimat))
        #
        leftprefix = vmat
        mps_dict[idx] = aten
        s_l_1 = s_l
    #到最后一个的时候，剩下的是这个mps的长度
    norm = leftprefix[0, 0]
    print(mps_dict, leftprefix)
    #
    ret = MPSState(mps_dict[1], mps_dict[length], length)
    for idx in range(2, length):
        ret.add_site(mps_dict[idx])
    return ret, norm


def create_random_mps_with_pnum(length, maxd):
    '''构造一个随机的mps，这个mps中的A满足粒子数守恒'''
    a_l_1 = 1
    a_l = None
    parti_nums = numpy.array([[0, 0]])
    #print(parti_nums.shape)
    delta_nums = numpy.array([(0, 0), (1, 0), (0, 1), (1, 1)])
    #
    mps_dict = {}
    #所有可能的粒子数的配置
    a_l_pool = []
    for leftleg in range(a_l_1):
        for idx in range(4):
            a_l_pool.append(parti_nums[leftleg] + delta_nums[idx])
    parti_pool = numpy.array(a_l_pool)
    #print(parti_pool.shape)
    #print(parti_pool)
    for idx in range(1, length+1):
        #首先确定a_l的数值是多大
        rightdim = numpy.power(4, length-idx)
        a_l = 4 * a_l_1
        a_l = numpy.min([rightdim, a_l, maxd])
        #随机生成一个张量
        ranten = numpy.random.randn(a_l_1, 4, a_l)
        #然后每个a_l随机选一个粒子数的配置，再把不是的值设置成0
        a_l_pnum_list = []
        for a_l_idx in range(a_l):
            a_l_pnum = parti_pool[numpy.random.randint(0, len(parti_pool))]
            for leftleg in range(a_l_1):
                for topleg in range(4):
                    left_top = parti_nums[leftleg] + delta_nums[topleg]
                    if not numpy.allclose(a_l_pnum, left_top):
                        ranten[leftleg, topleg, a_l_idx] = 0
            a_l_pnum_list.append(a_l_pnum)
        #然后更新现在的parti_num
        parti_nums = numpy.array(a_l_pnum_list)
        #
        mps_dict[idx] = MTensor(a_l_1, a_l, idx, ranten, parti_nums)
        #更新循环需要的变量
        a_l_1 = a_l
        #所有可能的粒子数的配置
        a_l_pool = []
        for leftleg in range(a_l_1):
            for topleg in range(4):
                a_l_pool.append(parti_nums[leftleg] + delta_nums[topleg])
        parti_pool = numpy.array(a_l_pool)
        #print(parti_pool.shape)
        #print(parti_pool)
    ret = MPSState(mps_dict[1], mps_dict[length], length)
    for idx in range(2, length):
        ret.add_site(mps_dict[idx])
    return ret
