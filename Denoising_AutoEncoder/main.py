# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
from data import standard_scale,get_random_block_from_data
from model import AdditiveGaussianNoiseAutoEncoder
import tensorflow as tf
import os

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
#对训练集和测试集进行标准化变换
X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)
n_samples=int(mnist.train.num_examples) #总训练样本数
training_epochs=30
batch_size=640
display_step=1 #每隔一轮就显示一次损失cost
#创建AGN自编码器的实例，
autoencoder=AdditiveGaussianNoiseAutoEncoder(n_input=784,n_hidden=600,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)
#一轮训练，可尝试修改参数来获得更低的cost
print('..............................train.................................')
print('..............................train.................................')
for epoch in range(training_epochs):
    avg_cost=0 #平均损失
    total_batch=int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs=get_random_block_from_data(X_train,batch_size)
        cost=autoencoder.partial_fit(batch_xs) #训练数据并计算当前cost
        avg_cost+=cost/n_samples*batch_size
    if epoch%display_step==0: #此处因display_step是1，故每一轮都会输出cost
        print('Epoch:','%04d'%(epoch+1),'cost=','{:.9f}'.format(avg_cost))
#对训练完的模型进行性能测试,测试集
print('..............................test.................................')
print('..............................test.................................')
print("Total cost:"+str(autoencoder.calc_total_cost(X_test))) #评价指标是平方误差