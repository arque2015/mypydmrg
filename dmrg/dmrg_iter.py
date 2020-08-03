"""
包含DMRG的sweep的函数
"""

from dmrghelpers.superblockhelper import extend_merge_to_superblock
from dmrghelpers.superblockhelper import leftext_hamiltonian_to_superblock
from dmrghelpers.superblockhelper import rightext_hamiltonian_to_superblock
from dmrghelpers.superblockhelper import leftext_oper_to_superblock
from dmrghelpers.superblockhelper import rightext_oper_to_superblock
from dmrghelpers.hamhelper import plus_two_hamiltonian
from .storages import BlockStorage

def get_superblock_ham(
        leftstorage: BlockStorage,
        rightstorage: BlockStorage,
        spin_sector,
        bonds
    ):
    '''获得superblock上的哈密顿量，这个时候需要指定sector\n
    '''
    leftext = leftstorage.block
    leftham = leftstorage.hamiltonian
    rightext = rightstorage.block
    rightham = rightstorage.hamiltonian
    superext = extend_merge_to_superblock(leftext, rightext)
    print(superext)
    #把左边的哈密顿量扩展到superblock
    lefthamext = leftext_hamiltonian_to_superblock(superext, leftham)
    #把右边的哈密顿量扩展到superblock
    righthamext = rightext_hamiltonian_to_superblock(superext, rightham)
    #把两个结果加起来
    superham = plus_two_hamiltonian(lefthamext, righthamext)
    del lefthamext
    del righthamext
    #把bond加起来
    op_dict = {}
    for bond in bonds:
        op1_idx, op2_idx = bond
        #找出第一个位置上的两个算符
        if op1_idx in op_dict:
            op1up, op1down = op_dict[op1_idx]
        else:
            op1up = leftstorage.get_oper(op1_idx, 1)
            op1up = leftext_oper_to_superblock(superext, op1up)
            op1down = leftstorage.get_oper(op1_idx, -1)
            op1down = leftext_oper_to_superblock(superext, op1down)
            op_dict[op1_idx] = (op1up, op1down)
        #找出第二个位置上的两个算符
        if op2_idx in op_dict:
            op2up, op2down = op_dict[op2_idx]
        else:
            op2up = rightstorage.get_oper(op2_idx, 1)
            op2up = rightext_oper_to_superblock(superext, op2up)
            op2down = rightstorage.get_oper(op2_idx, -1)
            op2down = rightext_oper_to_superblock(superext, op2down)
            op_dict[op2_idx] = (op2up, op2down)
        #把这个新的hopping项加进去
        superham.add_hopping_term(op1up, op2up)
        superham.add_hopping_term(op1down, op2down)
    #找到符合sector的所有idx
    sector_idxs = []
    for stcode in superext.iter_sitecode():
        lbkid, lstid, rstid, rbkid = stcode
        superidx = superext.sitecode_to_idx(stcode)
        lextidx = leftext.idxpair_to_idx(lbkid, lstid)
        rextidx = rightext.idxpair_to_idx(rstid, rbkid)
        nspin = leftext.spin_nums[lextidx] + rightext.spin_nums[rextidx]
        if nspin[0] == spin_sector[0] and nspin[1] == spin_sector[1]:
            sector_idxs.append(superidx)
    #
    print('len', len(sector_idxs))
    mat = superham.get_block(sector_idxs)
    return mat
