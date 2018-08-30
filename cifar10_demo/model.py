# coding=utf-8
import tensorflow as tf

#计算CNN的loss
def loss(logits,labels):
    labels=tf.cast(labels,tf.int64)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_per_example') #依然使用cross entropy，但把softmax的计算和cross entropy loss的计算合在一起
    cross_entropy_mean=tf.reduce_mean(cross_entropy,name='cross_entropy') #对cross entropy计算均值
    tf.add_to_collection('losses',cross_entropy_mean) #将cross entropy的loss添加到整体losses的collection中
    return tf.add_n(tf.get_collection('losses'),name='total_loss') #将整体losses的collection中的全部loss求和，得到最终的loss，包括cross entropy loss，还有后两个全连接层中weight的l2 loss.