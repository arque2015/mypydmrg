"""
包含需要计算的模型
自定义的模型需要继承BaseModel类
"""


class BaseModel(object):
    """规定晶格模型的基本功能"""
    def __init__(self, sites):
        self._sites = sites
        self._bonds = {}
        self._coef_u = 0.
        self._coef_mu = 0.
        self._coef_t = 1.
        #self._site_energy = None

    @property
    def size(self):
        '''模型的大小'''
        return len(self._sites)

    @property
    def sites(self):
        '''模型的格子编号'''
        return self._sites

    @property
    def coef_u(self):
        '''U'''
        return self._coef_u

    @property
    def coef_mu(self):
        '''Mu'''
        return self._coef_mu

    def get_site_bonds(self, stidx):
        '''获得一个site的所有bond\n
        继承了BaseModel的类可以重写这个方法来实现想要的厄效果
        '''
        return self._bonds[stidx]


    def get_t_coef(self, st1, st2):
        '''获得两个site之间的hopping的强度\n
        继承了BaseModel的类可以重写这个方法来实现t1,t2
        '''
        if st2 not in self.get_site_bonds(st1):
            raise ValueError('没有这个bond')
        return self._coef_t


    def __str__(self):
        template = '%s: \n' % self.__class__.__name__
        template += 'U: %.4f\n' % self._coef_u
        template += 'Mu: %.4f\n' % self._coef_mu
        for site in self._sites:
            template += '%d: %s\n' % (site, self._bonds[site])
        return template
