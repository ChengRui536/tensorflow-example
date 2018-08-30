# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
sess=tf.InteractiveSession() #创建一个TF默认的InteractiveSession,这样后面执行各项操作就无需指定Session了
#给隐含层的参数设置Variable并进行初始化
in_units=784 #输入节点数
h1_units=300 #隐含层的输出节点数
w1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1)) #隐含层的权重初始化为截断的正态分布，标准差为0.1，通过tf.truncated_normal()实现
b1=tf.Variable(tf.zeros([h1_units])) #隐含层的偏置
w2=tf.Variable(tf.zeros([h1_units,10])) #输出层的softmax，权重
b2=tf.Variable(tf.zeros([10])) #偏置
#定义输入x的placeholder
x=tf.placeholder(tf.float32,[None,in_units])
keep_prob=tf.placeholder(tf.float32) #因在训练和预测时，Dropout的比率keep_prob(即保留节点的概率)是不一样的，训练时小于1，预测时等于1，故把Dropout的比率作为计算图的输入
#定义模型结构
#隐含层
hidden1=tf.nn.relu(tf.matmul(x,w1)+b1) #实现一个激活函数为RuLU的隐含层
hidden1_drop=tf.nn.dropout(hidden1,keep_prob) #实现dropout的功能，即随机将一部分节点置为0
#输出层
y=tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)

#定义损失函数和选择优化器
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1])) #交叉信息熵
train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy) #学习速率0.3

#训练
tf.global_variables_initializer().run()
for i in range(3000): #一共30万的样本，相当于对全训练数据集进行了5轮epoch迭代
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75}) #保留75%的节点

#对模型进行准确率评测
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})) #0.981
print(accuracy.eval({x:mnist.validation.images,y_:mnist.validation.labels,keep_prob:1})) #0.9792