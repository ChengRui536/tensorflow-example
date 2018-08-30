# coding=utf-8
#卷积神经网络 两个卷积层加一个全连接层
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#权重初始化函数
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1) #给权重制造一些随机的噪声，截断的正态分布噪声，标准差设为0.1
    return tf.Variable(initial)

#偏置初始化函数
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape) #给偏置增加一些小的正值(0.1)用来避免死亡节点
    return tf.Variable(initial)

#卷积层函数
def conv2d(x,W): #x是输入，W是卷积的参数
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME') #二维卷积函数，strides表示卷积模版移动的步长，padding表示边界的处理方式，使卷积的输出和输入保持同样的尺寸

#池化层函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #2x2的最大池化

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
sess=tf.InteractiveSession()

#模型定义
x=tf.placeholder(tf.float32,[None,784]) #特征
y_=tf.placeholder(tf.float32,[None,10]) #真实的label
x_image=tf.reshape(x,[-1,28,28,1]) #因CNN会利用空间结构信息，将1D的输入向量转换为2D的图片结构，-1表示样本数量不固定，1代表颜色通道数量
#定义第一个卷积层
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
#定义第二个卷积层
W_conv2=weight_variable([5,5,32,64]) #卷积核的数量变为64，第二层卷积会提取64种特征
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
#将2D图片结构转换为1D向量结构，并连接一个全连接层
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64]) #对第二个卷积层的输出tensor进行变形，转换为1D向量
#全连接层
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1) #隐含节点1024，使用relu激活函数
#dropout层，减轻过拟合
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#softmax层
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#定义损失函数和优化器
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #给予一个较小的学习速率1e-4

#定义评测准确率
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#训练
tf.global_variables_initializer().run() #初始化所有参数
for i in range(20000):
    batch=mnist.train.next_batch(50)
    if i%100==0: #训练中评测时的keep_prob设为1，实时监测模型的性能
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

#在测试集和验证集上进行测试
# print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})) #10000个样本，'std::bad_alloc'错误，可以使用mnist.test.next_batch取其中一部分数据进行测试
print("validation accuracy %g"%accuracy.eval(feed_dict={x:mnist.validation.images,y_:mnist.validation.labels,keep_prob:1.0})) #0.9928