"""
测试operhelper.py的功能
"""

#import numpy
from dmrghelpers.blockhelper import first_leftblock, extend_leftblock
from dmrghelpers.blockhelper import update_to_leftblock
from dmrghelpers.operhelper import OperFactory
from dmrghelpers.operhelper import create_operator_of_site
from dmrghelpers.operhelper import leftblock_extend_oper
from dmrghelpers.operhelper import leftsite_extend_oper
from dmrghelpers.operhelper import update_leftblockextend_oper
from dmrghelpers.blockhelper import first_rightblock, extend_rightblock
from dmrghelpers.blockhelper import update_to_rightblock
from dmrghelpers.operhelper import rightblock_extend_oper, rightsite_extend_oper
from dmrghelpers.operhelper import update_rightblockextend_oper
from testhelper import random_phival

def test1():
    '''测试leftblock'''
    #first_block中的phi有四个分量，内容和一个格子是一致的
    left = first_leftblock(1)
    cuop = create_operator_of_site(left.fock_basis, OperFactory.create_spinup())
    #nuop = create_operator_of_site(left.fock_basis, OperFactory.number_spinup())
    leftext = extend_leftblock(left)
    print(left)
    print(cuop)
    #print(nuop)
    #print('验证：', numpy.matmul(cuop.mat, cuop.mat.transpose()))
    cuopext = leftblock_extend_oper(leftext, cuop)
    #nuopext = leftblock_extend_oper(leftext, nuop)
    print(cuopext)
    #print('验证2：', nuopext)
    #print(numpy.matmul(cuopext.mat,cuopext.mat.transpose()))
    #
    print('site上的算符更新')
    #
    cuop2 = create_operator_of_site(leftext.stbss, OperFactory.create_spinup())
    #nuop2 = create_operator_of_site(leftext.stbss, OperFactory.number_spinup())
    print(cuop2)
    cuop2ext = leftsite_extend_oper(leftext, cuop2)
    #nuop2ext = leftsite_extend_oper(leftext, nuop2)
    print(cuop2ext)
    #print('验证3：', nuop2ext)
    #print(numpy.matmul(cuop2ext.mat, cuop2ext.mat.transpose()))
    #
    phival = random_phival([4, 16], leftext)
    newleft = update_to_leftblock(leftext, phival)
    print(newleft)
    #
    newcuop = update_leftblockextend_oper(newleft, cuopext, phival)
    #newnuop = update_leftblockextend_oper(newleft, nuopext, phival)
    print(newcuop)
    #print('验证4', newnuop)
    #print(numpy.matmul(newnuop.mat, newnuop.mat.transpose()))
    #
    newleftext = extend_leftblock(newleft)
    newcuopext = leftblock_extend_oper(newleftext, newcuop)
    phival = random_phival([8, 16], newleftext)
    newnewleft = update_to_leftblock(newleftext, phival)
    newnewcuop = update_leftblockextend_oper(newnewleft, newcuopext, phival)
    print(newnewcuop)


def test2():
    '''测试right相关的'''
    right = first_rightblock(10)
    cudown = create_operator_of_site(right.fock_basis, OperFactory.create_spindown())
    print(cudown)
    #扩展right
    rightext = extend_rightblock(right)
    cuup2 = create_operator_of_site(rightext.stbss, OperFactory.create_spinup())
    #扩展rightext上面的两个算符
    cudownext = rightblock_extend_oper(rightext, cudown)
    print(cudownext)
    cuup2ext = rightsite_extend_oper(rightext, cuup2)
    print(cuup2ext)
    #将扩展完的两个算符收缩到新的right上
    phival = random_phival([16, 16], rightext)
    newright = update_to_rightblock(rightext, phival)
    print(newright)
    #print(newright.fock_dict)
    newcudown = update_rightblockextend_oper(newright, cudownext, phival)
    newcuup2 = update_rightblockextend_oper(newright, cuup2ext, phival)
    print(newcudown)
    print(newcuup2)
    #再给right扩展一个
    newrightext = extend_rightblock(newright)
    print(newrightext)#, newrightext._fock_basis)
    #扩展newright上面的算符
    newcudownext = rightblock_extend_oper(newrightext, newcudown)
    phival = random_phival([8, 64], newrightext)
    newnewright = update_to_rightblock(newrightext, phival)
    newnewcudown = update_rightblockextend_oper(newnewright, newcudownext, phival)
    print(newnewcudown)
    #print(newnewright._fock_dict[0])

def main():
    '''开始测试'''
    #test1()
    test2()


if __name__ == "__main__":
    main()
