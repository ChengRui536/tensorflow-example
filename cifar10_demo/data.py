# coding=utf-8
import tensorflow as tf

#定义权重初始化函数
def variable_with_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev)) #使用截断的正态分布来初始化权重
    #使用w1控制l2 loss的大小，w1为weight loss
    if w1 is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),w1,name='weight_loss') #给weight加l2的loss，即做一个l2的正则化处理，tf.nn.l2_loss计算weight的l2 loss
        tf.add_to_collection('losses',weight_loss) #losses会在后面计算NN的总体loss时用上
    return var