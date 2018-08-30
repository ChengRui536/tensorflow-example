# coding=utf-8
import cifar10,cifar10_input
import tensorflow as tf
from data import variable_with_weight_loss
from model import loss
import time,math
import numpy as np

max_steps=3000 #训练轮数
batch_size=128
data_dir='/tmp/cifar10_data/cifar-10-batches-bin' #下载cifar10数据的默认路径
#使用cifar10类下载数据集，并解压、展开到其默认位置
cifar10.maybe_download_and_extract()
#生成训练数据，在此处对数据进行了数据增强(翻转、裁剪、亮度和对比度)，并对数据进行了标准化，返回是已经封装好的tensor，每次执行生成一个batch_size的样本
images_train,labels_train=cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
#生成测试数据，进行了数据裁剪和标准化
images_test,labels_test=cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)

#定义模型架构
#创建输入数据
image_holder=tf.placeholder(tf.float32,[batch_size,24,24,3]) #样本条数，裁剪后图片尺寸24x24，颜色通道数3
labels_holder=tf.placeholder(tf.int32,[batch_size])

#特征提取工作
#第一个卷积层
#权重初始化操作
weight1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0) #创建卷积核的参数并进行初始化，5x5的卷积核大小，3个颜色通道，64个卷积核，weight初始化函数的标准差为0.05，因不对其weight进行l2正则，w1设为0
#卷积层
kernel1=tf.nn.conv2d(image_holder,weight1,[1,1,1,1],padding='SAME') #对输入数据image_holder进行卷积操作，步长strides均设为1
bias1=tf.Variable(tf.constant(0.0,shape=[64])) #bias全部初始化0
conv1=tf.nn.relu(tf.nn.bias_add(kernel1,bias1)) #将卷积的结果加上bias，使用relu()激活函数进行非线性化
#最大池化层
pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME') #使用一个尺寸为3x3，步长为2x2的最大池化层
#LRN层
norm1=tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75) #使用LRN对结果进行处理，增强模型的泛化能力
#第二个卷积层
#权重初始化操作
weight2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
#卷积层
kernel2=tf.nn.conv2d(norm1,weight2,[1,1,1,1],padding='SAME')
bias2=tf.Variable(tf.constant(0.1,shape=[64])) #bias全部初始化为0.1
conv2=tf.nn.relu(tf.nn.bias_add(kernel2,bias2))
#LRN层
norm2=tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
#最大池化层
pool2=tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#对特征进行组合匹配和分类
#第一个全连接层
#将第二个卷积层的输出结果扁平化
reshape=tf.reshape(pool2,[batch_size,-1]) #将每个样本都变成一维向量
dim=reshape.get_shape()[1].value #获取数据扁平化之后的长度
#权重初始化操作
weight3=variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004) #隐含节点数384，正态分布的标准差0.04。不希望这个全连接层过拟合，设weight loss0.004，故这一层的所有参数都被l2正则所约束
#全连接层
bias3=tf.Variable(tf.constant(0.1,shape=[384])) #bias全部初始化为0
local3=tf.nn.relu(tf.matmul(reshape,weight3)+bias3) #使用relu()激活函数进行非线性化
#第二个全连接层
#权重初始化操作
weight4=variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004) #隐含节点数下降了一半-192
#全连接层
bias4=tf.Variable(tf.constant(0.1,shape=[192]))
local4=tf.nn.relu(tf.matmul(local3,weight4)+bias4)
#最后一层
weight5=variable_with_weight_loss(shape=[192,10],stddev=1/192.0,w1=0.0) #不计入l2正则
bias5=tf.Variable(tf.constant(0.0,shape=[10]))
logits=tf.add(tf.matmul(local4,weight5),bias5)

#计算CNN的loss
loss=loss(logits,labels_holder) #得到最终loss

#优化器设置，学习速率设为1e-3，最小化loss
train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)

#输出准确率
top_k_op=tf.nn.in_top_k(logits,labels_holder,1) #求输出结果中top k的准确率，输出分数最高的那一类的准确率


#创建默认session并初始化全部模型参数
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
#启动图片数据增强的线程队列，共使用了16个线程来进行加速
tf.train.start_queue_runners()

#在训练集上训练
for step in range(max_steps):
    start_time=time.time()
    image_batch,label_batch=sess.run([images_train,labels_train]) #获得一个batch的训练数据
    _,loss_value=sess.run([train_op,loss],feed_dict={image_holder:image_batch,labels_holder:label_batch}) #train_op,loss的计算
    duration=time.time()-start_time
    if step%10==0: #每10个轮回
        examples_per_sec=batch_size/duration
        sec_per_batch=float(duration)
        format_str=('step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)')
        print(format_str % (step,loss_value,examples_per_sec,sec_per_batch)) #计算并展示当前loss，每秒钟能训练的样本数量，训练一个batch数据花费的时间

#在测试集上测试
num_examples=10000
num_iter=int(math.ceil(num_examples/batch_size))
true_count=0
total_sample_count=num_iter*batch_size
step=0
while step<num_iter:
    image_batch,label_batch=sess.run([images_test,labels_test]) #获得一个batch的训练数据
    predictions=sess.run([top_k_op],feed_dict={image_holder:image_batch,labels_holder:label_batch}) #计算模型在这个batch上的top 1上预测正确的样本数，不太明白top_k_op这个函数？？？？？？？？？？？？？？？？
    true_count+=np.sum(predictions) #所有预测正确的样本数
    step+=1
precision=true_count/total_sample_count
print('precision @ 1 = %.3f'%precision)
