# coding=utf-8
import tensorflow as tf
from others import print_activations

#AlexNet的网络结构
def inference(images,batch_size):
    parameters=[]
    #第1个卷积层
    with tf.name_scope('conv1') as scope: #将scope内生成的Variable自动命名为conv1/xxx，便于区分不同卷积层之间的组件
        #卷积层
        kernel=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=1e-1),name='weights') #使用截断的正态分布(标准差0.1)函数初始化卷积核的参数kernel，卷积核尺寸11x11，颜色通道3，卷积核数量64
        conv=tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME') #对输入image完成卷积操作，strides步长4x4
        biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name='biases') #biases全部初始化0
        bias=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(bias,name=scope) #使用激活函数对结果进行非线性处理
        print_activations(conv1)
        parameters+=[kernel,biases] #将这一层可训练的参数kernel，biases添加到parameters中
    #LRN层
    lrn1=tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1') #depth_radius为4，AlexNet论文中的推荐值
    #最大池化层
    pool1=tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1') #池化尺寸3x3，取样步长2x2，padding模式VALID，即取样时不能超过边框
    print_activations(pool1)

    #第2个卷积层，卷积核数量增加
    with tf.name_scope('conv2') as scope:
        #卷积层
        kernel=tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=1e-1),name='weights') #卷积核尺寸5x5
        conv=tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME') #卷积步长全设为1
        biases=tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
        print_activations(conv2)
    #LRN层
    lrn2=tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn2')
    #最大池化层
    pool2=tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')
    print_activations(pool2)

    #第3个卷积层，卷积核数量增加
    with tf.name_scope('conv3') as scope:
        #卷积层
        kernel=tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv3=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
        print_activations(conv3)

    #第4个卷积层，卷积核数量下降
    with tf.name_scope('conv4') as scope:
        #卷积层
        kernel=tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv4=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
        print_activations(conv4)

    #第5个卷积层，卷积核数量不变
    with tf.name_scope('conv5') as scope:
        #卷积层
        kernel=tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv5=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
        print_activations(conv5)
    #最大池化层
    pool5=tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool5')
    print_activations(pool5)

    #第1个全连接层，隐含节点数4096
    reshape=tf.reshape(pool5,[batch_size,-1])
    dim=reshape.get_shape()[1].value
    weight1=tf.Variable(tf.truncated_normal([dim,4096],stddev=0.1))
    bias1=tf.Variable(tf.constant(0.0,shape=[4096]))
    local1=tf.nn.relu(tf.matmul(reshape,weight1)+bias1)

    #第2个全连接层，隐含节点数4096

    #第3个全连接层，隐含节点数1000

    return pool5,parameters

