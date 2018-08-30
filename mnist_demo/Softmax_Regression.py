# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
# print(mnist)
# print(mnist.train.images.shape,mnist.train.labels.shape)
# print(mnist.test.images.shape,mnist.test.labels.shape)
# print(mnist.validation.images.shape,mnist.validation.labels.shape)

#定义分类算法(首先得知道算法的计算公式)
sess=tf.InteractiveSession() #将这个session注册为默认的session，之后的运算也默认跑在这个session中
x=tf.placeholder(tf.float32,[None,784]) #placeholder是输入数据的地方，第二个参数是tensor的shape,即数据的尺寸，None代表不限条数的输入
#创建weights和biases的Variable对象，全部初始化为0，模型训练时会自动学习合适的值
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
#实现Sotfmax Regression算法
y=tf.nn.softmax(tf.matmul(x,w)+b)

#定义损失函数
y_=tf.placeholder(tf.float32,[None,10])  #输入是真实的lable
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1])) #对每个batch数据结果求均值

#定义优化器
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #迭代执行训练操作train_step，

tf.global_variables_initializer().run()
#迭代执行训练操作train_step
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100) #随机从训练集中抽出100条样本构成mini-batch
    train_step.run({x:batch_xs,y_:batch_ys}) #将数据feed给placeholder,调用train_step进行训练

#对模型的准确率进行验证
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#统计全部样本预测的accuracy
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #将correct_prediction输出的bool转换成float32,再求平均
#输入测试数据的 特征和label 至评测流程accuracy，计算模型在测试集和验证集上的准确率
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels})) #0.9158
print(accuracy.eval({x:mnist.validation.images,y_:mnist.validation.labels})) #0.9206