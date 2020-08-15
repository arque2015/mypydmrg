"""和MPS有关的内容"""

import numpy

class MTensor():
    '''表示一个M，有三个指标，Delta_l，a_l-1还有a_l\n
    a_l_1,a_l这里面的参数分别是指定一共有多少维度的\n
    而idx指定了l是多少，每个格子上四种状态是固定的，分别是o,u,d,f\n
    '''
    def __init__(
            self,
            a_l_1, a_l,
            idx, entity,
            pnums=None#每个a_l包含的粒子数
        ):
        #
        self._pnums = pnums
        #将数据保存
        if isinstance(entity, numpy.ndarray):
            self._entity = numpy.reshape(entity, [a_l_1, 4, a_l])
        else:
            self._entity = numpy.array(entity)
            self._entity = numpy.reshape(self._entity, [a_l_1, 4, a_l])
        self._tshape = numpy.shape(self._entity)
        self._idx = idx

    @property
    def idx(self):
        '''这个tensor对应的格子编号'''
        return self._idx

    @property
    def tensor_shape(self):
        '''tensor的shape，delta，al_1, al'''
        return self._tshape

    @property
    def tensor_entity(self):
        '''tensor的内容'''
        return self._entity

    def delta_of(self, delta):
        '''获取某个delta下的(a_l-1, al)矩阵'''
        #print(self._entity.shape)
        return self._entity[:, delta, :]


    def __str__(self):
        idx = self._idx
        shape = self._tshape
        template = '%s: \n' % self.__class__.__name__
        template += 'M^{delta_%d}_{a_%d,a_%d}\n' % (idx, idx-1, idx)
        template += '\ta_%d=%d' % (idx-1, shape[0])
        template += '\ta_%d=%d\n' % (idx, shape[2])
        if self._pnums is not None:
            template += str(self._pnums)
        template += '\n'
        return template


    def __repr__(self):
        return self.__str__()

class MPSContainer():
    '''包含一系列的张量\n
    想要得到一个
    '''
    def __init__(self, tensor: MTensor):
        self._delta_dim = tensor.tensor_shape[1]
        self._container = {}
        self._container[tensor.idx] = tensor

    def left_iter(self):
        '''从左到右进行遍历'''
        idxlist = numpy.sort(list(self._container.keys()))
        for idx in idxlist:
            yield idx, self._container[idx]

    def add_site(self, tensor):
        '''添加一个site'''
        if tensor.idx in self._container:
            raise ValueError('这个格子已经存在了')
        self._container[tensor.idx] = tensor

    def left_shrink(self, sitecode):
        '''从小idx收缩到大idx'''
        idxlist = numpy.sort(list(self._container.keys()))
        mat = numpy.eye(1)#self._container[idxlist[0]][sitecode[0]]
        for idx, code in zip(idxlist, sitecode):
            #print(idx, code)
            #print(self._container[idx].delta_of(code))
            #print(numpy.shape(mat), numpy.shape(self._container[idx].delta_of(code)))
            mat = numpy.matmul(mat, self._container[idx].delta_of(code))
        #print(mat)
        return mat


    def __str__(self):
        template = '%s: \n' % self.__class__.__name__
        template += 'delta dim: %d, length: %d\n' %\
            (self._delta_dim, len(self._container))
        idxlist = numpy.sort(list(self._container.keys()))
        for idx in idxlist:
            template += str(self._container[idx])
        return template
