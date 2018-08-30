# coding=utf-8

#显示网络每一层结构的函数
def print_activations(t): #t是一个tensor
    print(t.op.name,' ',t.get_shape().as_list())