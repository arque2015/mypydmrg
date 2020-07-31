"""
辅助测试的一些函数
"""

import numpy

def random_phival(shape, lorrext):
    '''生成随机的phival用来updateblock'''
    phival = numpy.zeros(shape)#比如保留8个
    #找出有相同的spin_num的basis
    samepnumdic = {}
    for idx in range(lorrext.dim):
        spnum = lorrext.spin_nums[idx]
        #区分高位和低位，可能的粒子数从0到block数
        spidx = spnum[1]*(lorrext.block_len+1) + spnum[0]
        if not spidx in samepnumdic:
            samepnumdic[spidx] = []
        samepnumdic[spidx].append(idx)
    #
    #随机生成一个phival
    #print(samepnumdic)
    keys = list(samepnumdic.keys())
    for alpha in range(shape[0]):
        #先随机挑选一个sector
        sector = numpy.random.randint(0, len(keys))
        sector = keys[sector]#pylint: disable=invalid-sequence-index
        #随机生成分量
        ranvec = numpy.random.rand(len(samepnumdic[sector]))
        dotlen = numpy.dot(ranvec, ranvec)
        ranvec = ranvec / numpy.sqrt(dotlen)
        #
        for idx2, val in zip(samepnumdic[sector], ranvec):
            phival[alpha, idx2] = val
    #print(phival)
    return phival
