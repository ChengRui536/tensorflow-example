# coding=utf-8
import tensorflow as tf
from data import xavier_init
import numpy as np


class AdditiveGaussianNoiseAutoEncoder(object):
    #神经网络的设计，输入变量数、隐含层节点数、隐含层激活函数、优化器、高斯噪声系数，使用了一个隐含层
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.transfer=transfer_function
        self.scale=tf.placeholder(tf.float32)
        self.training_scale=scale
        network_weights=self._initialize_weights() #参数初始化，_initialize_weights()
        self.weights=network_weights

        #定义网络结构
        self.x=tf.placeholder(tf.float32,[None,self.n_input]) #输入x
        #能提取特征的隐含层，学习出数据中的高阶特征，给输入x加上噪声self.x+scale*tf.random_normal((n_input,))，噪声输入与输入层权重w1相乘，加上隐含层的偏置b1，最后对结果进行激活函数处理
        self.hidden1=self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input,)),self.weights['w1']),self.weights['b1']))
        self.hidden2=self.transfer(tf.add(tf.matmul(self.hidden1+scale*tf.random_normal((n_input,)),self.weights['w1']),self.weights['b1']))
        #在输出层进行数据复原和重建操作，将隐含层的输出结果乘输出层的权重w2，再加上输出层的偏置b2
        self.reconstruction=tf.add(tf.matmul(self.hidden2,self.weights['w2']),self.weights['b2'])

        #定义损失函数，直接使用平方误差
        #先计算输出self.reconstruction与输入self.x之差，后求差的平方，最后求和得到平方误差
        self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        #定义优化器，其实就是训练过程
        self.optimizer=optimizer.minimize(self.cost)

        init=tf.global_variables_initializer() #初始化自编码器的全部模型参数
        self.sess=tf.Session() #创建Session
        self.sess.run(init)


    #权重初始化函数
    def _initialize_weights(self):
        all_weights=dict()
        all_weights['w1']=tf.Variable(xavier_init(self.n_input,self.n_hidden)) #返回一个比较适合于激活函数softplus的权重初始分布
        all_weights['b1']=tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32)) #偏置b1全部置为0
        all_weights['w2']=tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32)) #输出层因没有使用激活函数，全部初始化为0即可
        all_weights['b2']=tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights


    #用一个batch数据进行一步训练，触发训练操作并返回当前的损失cost，optimizer是训练过程
    def partial_fit(self,X):
        cost,optim=self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X,self.scale:self.training_scale})
        return cost


    #只求损失cost，在自编码器训练完成后，在测试集上对模型性能进行评测
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})


    # #返回自编码器隐含层的输出结果，提供一个接口来获取抽象后的特征，计算图中的子图
    # def transfrom(self,X):
    #     return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})
    #
    #
    # #以隐含层的输出结果作为输入，用重建层将高阶特征复原为原始数据，计算图中的子图
    # def generate(self,hidden=None):
    #     if hidden is None:
    #         hidden=np.random.normal(size=self.weights['b1'])
    #     return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})
    #
    #
    # #整体运行一遍复原过程，包括提取高阶特征过程和通过高阶特征复原数据，即包括transfrom和generate两块
    # #输入数据是原始数据，输出数据是复原后的数据
    # def reconstruct(self,X):
    #     return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})
    #
    #
    # #获取隐含层的权重w1
    # def getWeights(self):
    #     return self.sess.run(self.weights['w1'])
    #
    #
    # #获取隐含层的偏置系数b1
    # def getBiases(self):
    #     return self.sess.run(self.weights['b1'])