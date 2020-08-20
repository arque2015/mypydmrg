"""创建superblock上用来观测的算符"""

from basics.basis import SiteBasis
from fermionic.baseop import BaseOperator
from .operhelper import OperFactory
from .superblockhelper import leftext_oper_to_superblock


MEASOP_DICT = {
    'sz': OperFactory.create_sz
}

def create_meas_operator_of_site(basis: SiteBasis, prefix):
    """创建观测对应的算符"""
    if len(basis.sites) > 1:
        raise ValueError('只能有一个格子')
    siteidx = basis.sites[0]
    tup = MEASOP_DICT[prefix]()
    return BaseOperator(siteidx, basis, tup.isferm, tup.mat, spin=tup.spin)


def get_leftext_blockmat_from_superidx(superext, meas, superidxs):
    """从leftext基上的算符中提取相应的块出来"""
    meassuper = leftext_oper_to_superblock(superext, meas)
    return meassuper.get_block(superidxs)

