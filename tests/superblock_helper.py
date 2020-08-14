"""
测试superblock中的有关功能
"""

import time
import numpy
from dmrghelpers.blockhelper import first_leftblock, first_rightblock
from dmrghelpers.blockhelper import extend_leftblock, extend_rightblock
from dmrghelpers.blockhelper import update_to_leftblock, update_to_rightblock
from dmrghelpers.operhelper import OperFactory
from dmrghelpers.operhelper import create_operator_of_site
from dmrghelpers.operhelper import leftsite_extend_oper, leftblock_extend_oper
from dmrghelpers.operhelper import leftsite_extend_oper_
from dmrghelpers.operhelper import rightsite_extend_oper, rightblock_extend_oper
from dmrghelpers.operhelper import rightblock_extend_oper_
from dmrghelpers.operhelper import update_leftblockextend_oper
from dmrghelpers.operhelper import update_rightblockextend_oper
from dmrghelpers.hamhelper import create_hamiltonian_of_site
from dmrghelpers.hamhelper import extend_leftblock_hamiltonian, extend_rightblock_hamiltonian
from dmrghelpers.hamhelper import update_leftblockextend_hamiltonian
from dmrghelpers.hamhelper import update_rightblockextend_hamiltonian
from dmrghelpers.hamhelper import plus_two_hamiltonian
from dmrghelpers.superblockhelper import extend_merge_to_superblock
from dmrghelpers.superblockhelper import leftext_hamiltonian_to_superblock
from dmrghelpers.superblockhelper import leftext_oper_to_superblock
from dmrghelpers.superblockhelper import rightext_hamiltonian_to_superblock
from dmrghelpers.superblockhelper import rightext_oper_to_superblock


def test_clock(func, *args):
    '''比较运行时间'''
    start = time.time()
    ret = func(*args)
    stop = time.time()
    print(func.__name__, stop - start)
    return ret

def test1():
    '''测试superblock
    测试在5个格子时候的哈密顿量是否正确，验证自旋上的部分
    '''
    #
    left1 = first_leftblock(1)
    #现在只有第一个格子，创建第一个格子的哈密顿量和产生算符
    hamleft = create_hamiltonian_of_site(left1.fock_basis, -1., 0)
    print('U', hamleft)
    cup_in_1 = create_operator_of_site(left1.fock_basis, OperFactory.create_spinup())
    #将第一个格子扩展到第二个
    left1ext = extend_leftblock(left1)
    #在扩展过的格子上的C^+_1
    cup_in_1 = leftblock_extend_oper(left1ext, cup_in_1)
    #扩展以后有C^+_2
    cup_in_2 = create_operator_of_site(left1ext.stbss, OperFactory.create_spinup())
    _cup_in_2_ = test_clock(leftsite_extend_oper_, left1ext, cup_in_2)
    cup_in_2 = test_clock(leftsite_extend_oper, left1ext, cup_in_2)
    print('对比leftsite_extend_oper', numpy.allclose(_cup_in_2_.mat, cup_in_2.mat))
    #扩展以后的U2
    ciu_in_2 = create_operator_of_site(left1ext.stbss, OperFactory.create_u())
    _ciu_in_2_ = test_clock(leftsite_extend_oper_, left1ext, ciu_in_2)
    ciu_in_2 = test_clock(leftsite_extend_oper, left1ext, ciu_in_2)
    print('对比leftsite_extend_oper', numpy.allclose(_ciu_in_2_.mat, ciu_in_2.mat))
    #现在C^+_1和C^+_2都在|phi^1, s^2>这个left1ext基上，整合进哈密顿量
    #先把哈密顿量也放到|phi^1, s^2>这个基上
    hamleft = extend_leftblock_hamiltonian(hamleft, left1ext)
    hamleft.add_hopping_term(cup_in_1, cup_in_2)
    hamleft.add_u_term(ciu_in_2, -1.0)
    print('U', hamleft)
    #然后升级left1ext到left2 |phi^2>
    phival = numpy.eye(16)
    left2 = update_to_leftblock(left1ext, phival)
    hamleft = update_leftblockextend_hamiltonian(left2, hamleft, phival)
    cup_in_1 = update_leftblockextend_oper(left2, cup_in_1, phival)
    cup_in_2 = update_leftblockextend_oper(left2, cup_in_2, phival)
    #将|phi^2>扩展到|phi^2, s^3>
    left2ext = extend_leftblock(left2)
    #把哈密顿量和第二个算符也扩展过去
    hamleft = extend_leftblock_hamiltonian(hamleft, left2ext)
    cup_in_2 = leftblock_extend_oper(left2ext, cup_in_2)
    #创建第三个格子的算符，然后放到|phi^2, s^3>
    cup_in_3 = create_operator_of_site(left2ext.stbss, OperFactory.create_spinup())
    _cup_in_3_ = test_clock(leftsite_extend_oper_, left2ext, cup_in_3)
    cup_in_3 = test_clock(leftsite_extend_oper, left2ext, cup_in_3)
    print('对比leftsite_extend_oper', numpy.allclose(_cup_in_3_.mat, cup_in_3.mat))
    #把2-3之间的hopping放到哈密顿量里
    hamleft.add_hopping_term(cup_in_2, cup_in_3)
    #把|phi^2, s^3>这个东西以后要用来生成supoerblock
    print(hamleft)
    print(cup_in_3)
    #
    #
    right = first_rightblock(5)
    #现在只有第5个格子
    hamright = create_hamiltonian_of_site(right.fock_basis, -1.0, 0)
    print('rU', hamright)
    cup_in_5 = create_operator_of_site(right.fock_basis, OperFactory.create_spinup())
    #将第5个格子扩展到第4个 |s^4, phi^1>
    rightext = extend_rightblock(right)
    #把第5个格子的算符扩展
    _cup_in_5_ = test_clock(rightblock_extend_oper_, rightext, cup_in_5)
    cup_in_5 = test_clock(rightblock_extend_oper, rightext, cup_in_5)
    print('对比rightblock_extend_oper', numpy.allclose(_cup_in_5_.mat, cup_in_5.mat))
    #创建第四个格子的算符并且扩展
    cup_in_4 = create_operator_of_site(rightext.stbss, OperFactory.create_spinup())
    cup_in_4 = rightsite_extend_oper(rightext, cup_in_4)
    #第四个格子上的U
    ciu_in_4 = create_operator_of_site(rightext.stbss, OperFactory.create_u())
    ciu_in_4 = rightsite_extend_oper(rightext, ciu_in_4)
    #把哈密顿量扩展到|s^4, phi^1>
    hamright = extend_rightblock_hamiltonian(hamright, rightext)
    #添加4-5的hopping
    hamright.add_hopping_term(cup_in_4, cup_in_5)
    hamright.add_u_term(ciu_in_4, -1.0)
    print('rU', hamright)
    print(cup_in_4)
    #
    #创建superblock
    superblock = extend_merge_to_superblock(left2ext, rightext)
    print(superblock)
    #把左边的哈密顿量扩展到superblock上面
    hamleft = leftext_hamiltonian_to_superblock(superblock, hamleft)
    cup_in_3 = leftext_oper_to_superblock(superblock, cup_in_3)
    print(hamleft)
    print(cup_in_3)
    #
    #for ridx in superblock.iter_idx():
    #    for lidx in superblock.iter_idx():
    #        if hamleft.mat[lidx, ridx] == 0:
    #            continue
    #        idx1, idx2, idx3, idx4 = superblock.idx_to_idxtuple(ridx)
    #        rsta = '%s,%s,%s,%s' %\
    #            (left1ext.idx_to_state(idx1),\
    #                superblock.leftblockextend.stbss.idx_to_state(idx2),\
    #                    superblock.rightblockextend.stbss.idx_to_state(idx3),\
    #                        superblock.rightblockextend.rblk.idx_to_state(idx4))
    #        idx1, idx2, idx3, idx4 = superblock.idx_to_idxtuple(lidx)
    #        lsta = '%s,%s,%s,%s' %\
    #            (left1ext.idx_to_state(idx1),\
    #                superblock.leftblockextend.stbss.idx_to_state(idx2),\
    #                    superblock.rightblockextend.stbss.idx_to_state(idx3),\
    #                        superblock.rightblockextend.rblk.idx_to_state(idx4))
    #        print(rsta, lsta, hamleft.mat[lidx, ridx])
    #for ridx in superblock.iter_idx():
    #    for lidx in superblock.iter_idx():
    #        if cup_in_3.mat[lidx, ridx] == 0:
    #            continue
    #        idx1, idx2, idx3, idx4 = superblock.idx_to_idxtuple(ridx)
    #        rsta = '%s,%s,%s,%s' %\
    #            (left1ext.idx_to_state(idx1),\
    #                superblock.leftblockextend.stbss.idx_to_state(idx2),\
    #                    superblock.rightblockextend.stbss.idx_to_state(idx3),\
    #                        superblock.rightblockextend.rblk.idx_to_state(idx4))
    #        idx1, idx2, idx3, idx4 = superblock.idx_to_idxtuple(lidx)
    #        lsta = '%s,%s,%s,%s' %\
    #            (left1ext.idx_to_state(idx1),\
    #                superblock.leftblockextend.stbss.idx_to_state(idx2),\
    #                    superblock.rightblockextend.stbss.idx_to_state(idx3),\
    #                        superblock.rightblockextend.rblk.idx_to_state(idx4))
    #        print(rsta, lsta, cup_in_3.mat[lidx, ridx])
    #
    #把右边的哈密顿量扩展到superblock上面
    hamright = rightext_hamiltonian_to_superblock(superblock, hamright)
    print(hamright)
    #for ridx in superblock.iter_idx():
    #    for lidx in superblock.iter_idx():
    #        if hamright.mat[lidx, ridx] == 0:
    #            continue
    #        idx1, idx2, idx3, idx4 = superblock.idx_to_idxtuple(ridx)
    #        rsta = '%s,%s,%s,%s' %\
    #            (left1ext.idx_to_state(idx1),\
    #                superblock.leftblockextend.stbss.idx_to_state(idx2),\
    #                    superblock.rightblockextend.stbss.idx_to_state(idx3),\
    #                        superblock.rightblockextend.rblk.idx_to_state(idx4))
    #        idx1, idx2, idx3, idx4 = superblock.idx_to_idxtuple(lidx)
    #        lsta = '%s,%s,%s,%s' %\
    #            (left1ext.idx_to_state(idx1),\
    #                superblock.leftblockextend.stbss.idx_to_state(idx2),\
    #                    superblock.rightblockextend.stbss.idx_to_state(idx3),\
    #                        superblock.rightblockextend.rblk.idx_to_state(idx4))
    #        print(rsta, lsta, hamright.mat[lidx, ridx])
    #
    #把右边的算符扩展到superblock上面
    cup_in_4 = rightext_oper_to_superblock(superblock, cup_in_4)
    print(cup_in_4)
    #for ridx in superblock.iter_idx():
    #    for lidx in superblock.iter_idx():
    #        if cup_in_4.mat[lidx, ridx] == 0:
    #            continue
    #        idx1, idx2, idx3, idx4 = superblock.idx_to_idxtuple(ridx)
    #        rsta = '%s,%s,%s,%s' %\
    #            (left1ext.idx_to_state(idx1),\
    #                superblock.leftblockextend.stbss.idx_to_state(idx2),\
    #                    superblock.rightblockextend.stbss.idx_to_state(idx3),\
    #                        superblock.rightblockextend.rblk.idx_to_state(idx4))
    #        idx1, idx2, idx3, idx4 = superblock.idx_to_idxtuple(lidx)
    #        lsta = '%s,%s,%s,%s' %\
    #            (left1ext.idx_to_state(idx1),\
    #                superblock.leftblockextend.stbss.idx_to_state(idx2),\
    #                    superblock.rightblockextend.stbss.idx_to_state(idx3),\
    #                        superblock.rightblockextend.rblk.idx_to_state(idx4))
    #        print(rsta, lsta, cup_in_4.mat[lidx, ridx])
    hamsuper = plus_two_hamiltonian(hamleft, hamright)
    hamsuper.add_hopping_term(cup_in_3, cup_in_4)
    print(hamsuper)
    #for ridx in superblock.iter_idx():
    #    for lidx in superblock.iter_idx():
    #        if hamsuper.mat[lidx, ridx] == 0:
    #            continue
    #        idx1, idx2, idx3, idx4 = superblock.idx_to_idxtuple(ridx)
    #        rsta = '%s,%s,%s,%s' %\
    #            (left1ext.idx_to_state(idx1),\
    #                superblock.leftblockextend.stbss.idx_to_state(idx2),\
    #                    superblock.rightblockextend.stbss.idx_to_state(idx3),\
    #                        superblock.rightblockextend.rblk.idx_to_state(idx4))
    #        idx1, idx2, idx3, idx4 = superblock.idx_to_idxtuple(lidx)
    #        lsta = '%s,%s,%s,%s' %\
    #            (left1ext.idx_to_state(idx1),\
    #                superblock.leftblockextend.stbss.idx_to_state(idx2),\
    #                    superblock.rightblockextend.stbss.idx_to_state(idx3),\
    #                        superblock.rightblockextend.rblk.idx_to_state(idx4))
    #        print(rsta, lsta, hamsuper.mat[lidx, ridx])

def test2():
    '''测试superblock
    测试在5个格子时候的哈密顿量是否正确，验证自旋下的部分
    '''
    #
    left1 = first_leftblock(1)
    #现在只有第一个格子，创建第一个格子的哈密顿量和产生算符
    hamleft = create_hamiltonian_of_site(left1.fock_basis, 0, 0)
    cdn_in_1 = create_operator_of_site(left1.fock_basis, OperFactory.create_spindown())
    #将第一个格子扩展到第二个
    left1ext = extend_leftblock(left1)
    #在扩展过的格子上的C^+_1
    cdn_in_1 = leftblock_extend_oper(left1ext, cdn_in_1)
    #扩展以后有C^+_2
    cdn_in_2 = create_operator_of_site(left1ext.stbss, OperFactory.create_spindown())
    _cdn_in_2_ = test_clock(leftsite_extend_oper_, left1ext, cdn_in_2)
    cdn_in_2 = test_clock(leftsite_extend_oper, left1ext, cdn_in_2)
    print('对比leftsite_extend_oper', numpy.allclose(_cdn_in_2_.mat, cdn_in_2.mat))
    #现在C^+_1和C^+_2都在|phi^1, s^2>这个left1ext基上，整合进哈密顿量
    #先把哈密顿量也放到|phi^1, s^2>这个基上
    hamleft = extend_leftblock_hamiltonian(hamleft, left1ext)
    hamleft.add_hopping_term(cdn_in_1, cdn_in_2)
    #然后升级left1ext到left2 |phi^2>
    phival = numpy.eye(16)
    left2 = update_to_leftblock(left1ext, phival)
    hamleft = update_leftblockextend_hamiltonian(left2, hamleft, phival)
    cdn_in_1 = update_leftblockextend_oper(left2, cdn_in_1, phival)
    cdn_in_2 = update_leftblockextend_oper(left2, cdn_in_2, phival)
    #将|phi^2>扩展到|phi^2, s^3>
    left2ext = extend_leftblock(left2)
    #把哈密顿量和第二个算符也扩展过去
    hamleft = extend_leftblock_hamiltonian(hamleft, left2ext)
    cdn_in_2 = leftblock_extend_oper(left2ext, cdn_in_2)
    #创建第三个格子的算符，然后放到|phi^2, s^3>
    cdn_in_3 = create_operator_of_site(left2ext.stbss, OperFactory.create_spindown())
    _cdn_in_3_ = test_clock(leftsite_extend_oper_, left2ext, cdn_in_3)
    cdn_in_3 = test_clock(leftsite_extend_oper, left2ext, cdn_in_3)
    print('对比leftsite_extend_oper', numpy.allclose(_cdn_in_3_.mat, cdn_in_3.mat))
    #把2-3之间的hopping放到哈密顿量里
    hamleft.add_hopping_term(cdn_in_2, cdn_in_3)
    #把|phi^2, s^3>这个东西以后要用来生成supoerblock
    print(hamleft)
    print(cdn_in_3)
    #
    #
    right = first_rightblock(5)
    #现在只有第5个格子
    hamright = create_hamiltonian_of_site(right.fock_basis, 0, 0)
    cdn_in_5 = create_operator_of_site(right.fock_basis, OperFactory.create_spindown())
    #将第5个格子扩展到第4个 |s^4, phi^1>
    rightext = extend_rightblock(right)
    #把第5个格子的算符扩展
    _cdn_in_5_ = test_clock(rightblock_extend_oper_, rightext, cdn_in_5)
    cdn_in_5 = test_clock(rightblock_extend_oper, rightext, cdn_in_5)
    print('对比rightblock_extend_oper', numpy.allclose(_cdn_in_5_.mat, cdn_in_5.mat))
    #创建第四个格子的算符并且扩展
    cdn_in_4 = create_operator_of_site(rightext.stbss, OperFactory.create_spindown())
    cdn_in_4 = rightsite_extend_oper(rightext, cdn_in_4)
    #把哈密顿量扩展到|s^4, phi^1>
    hamright = extend_rightblock_hamiltonian(hamright, rightext)
    #添加4-5的hopping
    hamright.add_hopping_term(cdn_in_4, cdn_in_5)
    print(hamright)
    print(cdn_in_4)
    #
    #创建superblock
    superblock = extend_merge_to_superblock(left2ext, rightext)
    print(superblock)
    #把左边的哈密顿量扩展到superblock上面
    hamleft = leftext_hamiltonian_to_superblock(superblock, hamleft)
    cdn_in_3 = leftext_oper_to_superblock(superblock, cdn_in_3)
    print(hamleft)
    print(cdn_in_3)
    #把右边的哈密顿量扩展到superblock上面
    hamright = rightext_hamiltonian_to_superblock(superblock, hamright)
    print(hamright)
    #把右边的算符扩展到superblock上面
    cdn_in_4 = rightext_oper_to_superblock(superblock, cdn_in_4)
    print(cdn_in_4)
    #
    hamsuper = plus_two_hamiltonian(hamleft, hamright)
    hamsuper.add_hopping_term(cdn_in_3, cdn_in_4)
    print(hamsuper)
    #for ridx in superblock.iter_idx():
    #    for lidx in superblock.iter_idx():
    #        if hamsuper.mat[lidx, ridx] == 0:
    #            continue
    #        idx1, idx2, idx3, idx4 = superblock.idx_to_idxtuple(ridx)
    #        rsta = '%s,%s,%s,%s' %\
    #            (left1ext.idx_to_state(idx1),\
    #                superblock.leftblockextend.stbss.idx_to_state(idx2),\
    #                    superblock.rightblockextend.stbss.idx_to_state(idx3),\
    #                        superblock.rightblockextend.rblk.idx_to_state(idx4))
    #        idx1, idx2, idx3, idx4 = superblock.idx_to_idxtuple(lidx)
    #        lsta = '%s,%s,%s,%s' %\
    #            (left1ext.idx_to_state(idx1),\
    #                superblock.leftblockextend.stbss.idx_to_state(idx2),\
    #                    superblock.rightblockextend.stbss.idx_to_state(idx3),\
    #                        superblock.rightblockextend.rblk.idx_to_state(idx4))
    #        print(rsta, lsta, hamsuper.mat[lidx, ridx])


def main():
    '''开始测试'''
    test1()
    test2()

if __name__ == "__main__":
    main()
