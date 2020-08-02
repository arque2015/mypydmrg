"""
运行dmrg的时候维护下来的变量等内容
"""

from lattice import BaseModel
from fermionic.baseop import BaseOperator, Hamiltonian

class DMRGConfig(object):
    """保存一些经常需要使用的值"""
    def __init__(
            self, 
            model: BaseModel,
            nrg_max_keep: int,
        ):
        self._model = model
        #一些算法控制的参数
        self.nrg_max_keep = nrg_max_keep
        #
        #block上面的算符个哈密顿量是用来向下一次进行递推的
        #临时存储就行了
        self._left_tmp = None
        self._right_tmp = None
        #多数运算都在superblock上面，保存extend以后的基更加方便
        #这里面存|phi^idx, s^idx+1>这个extend上面的算符
        #（右边是|s^idx-1, phi^(N-idx+1)>）
        self._leftext_storage = [BlockStorage(idx) for idx in range(model.size)]
        self._rightext_storage = [BlockStorage(idx) for idx in range(model.size)]


    def get_leftblock_storage(self, phi_idx):
        '''暂时保存的leftblock'''
        if phi_idx != self._left_tmp.phi_idx:
            raise ValueError('存储的idx和提取的idx不一致')
        return self._left_tmp

    def storage_leftext_ham(self, phi_idx, ham):
        '''存储一个ham进去\n
        这个算符应该再extend以后的基上，也就是|phi^idx, s^idx+1>\n
        这个要在以后的superblock上进行计算
        '''
        self._leftext_storage[phi_idx].storage_ham(ham)

    def storage_leftext_oper(self, phi_idx, oper):
        '''存储一个oper进去\n
        这个算符应该再extend以后的基上，也就是|phi^idx, s^idx+1>\n
        注意不要搞混了这个idx和oper的stidx，这个idx是block上面的指标\n
        不是算符在格子上的指标
        '''
        self._leftext_storage[phi_idx].storage_oper(oper)

    def storage_rightext_oper(self, phi_idx, oper):
        '''
        存储一个oper进去\n
        这个算符应该再extend以后的基上，也就是|s^idx-1, phi^(N-idx+1)>
        '''
        self._rightext_storage[phi_idx].storage_oper(oper)

    def left_tmp_reset(self, phi_idx, ham):
        '''清空left_tmp，需要制定idx还有哈密顿量\n
        这个在最开始的使用调用就可以了
        '''
        if self._left_tmp is not None:
            del self._left_tmp
        self._left_tmp = BlockStorage(phi_idx)
        self._left_tmp.storage_ham(ham)

    def left_tmp_add_oper(self, oper):
        '''增加一个leftblock上的算符，\n
        这个算符用来进行下一次的extend，是临时的'''
        self._left_tmp.storage_oper(oper)

    def __str__(self):
        template = 'DMRGConfig: \nModel: %s\n' % self._model.__class__.__name__
        if self._left_tmp is not None:
            template += str(self._left_tmp)
        return template


class BlockStorage(object):
    """每一个Block时，哈密顿量还有一些算符是以后也会用的\n
    需要保存下来
    """
    def __init__(self, idx):
        self._phi_idx = idx
        self._ham = None
        self._oper_storage = []
        self._opers = {}

    #@property
    #def oper_dict(self):
    #    '''包含的算符'''
    #    return self._opers

    @property
    def phi_idx(self):
        '''保存的内容的idx，可能是-block也可能是-blockextend\n
        注意这个idx是block的idx不是算符的
        '''
        return self._phi_idx

    #def inc_phi_idx(self):
    #    '''给phi_idx增加1
    #    '''
    #    self._phi_idx += 1

    @property
    def hamiltonian(self):
        '''保存的哈密顿量'''
        return self._ham

    @property
    def oper_storage(self):
        '''保存下来的算符的siteidx'''
        return self._oper_storage

    def storage_oper(self, oper: BaseOperator):
        '''向存储中增加一个oper'''
        key = '%d,%d' % (oper.siteidx, oper.spin)
        #因为有自旋上下两个部分，不能加重复了
        if not oper.siteidx in self._oper_storage:
            self._oper_storage.append(oper.siteidx)
        self._opers[key] = oper

    def get_oper(self, stidx, spin):
        '''从存储中拿出一个算符'''
        key = '%d,%d' % (stidx, spin)
        return self._opers[key]

    def pop_oper(self, stidx):
        '''从存储中拿出两个算符，并从存储中移出'''
        #从oper_storage中删除这个记录
        self._oper_storage.remove(stidx)
        up_key = '%d,%d' % (stidx, 1)
        down_key = '%d,%d' % (stidx, -1)
        #弹出两个算符
        up_op = self._opers.pop(up_key) if up_key in self._opers else None
        down_op = self._opers.pop(down_key) if down_key in self._opers else None
        return up_op, down_op

    def storage_ham(self, ham: Hamiltonian):
        '''存储对应的哈密顿量'''
        self._ham = ham


    def __str__(self):
        template = 'Storage: phi^%d\n' % self._phi_idx
        if self._ham is not None:
            template += 'ham: \n%s\n' % str(self._ham)
        else:
            template += 'ham: 未设置'
        for key in self._opers:
            template += '%s\n' % str(self._opers[key])
        return template
