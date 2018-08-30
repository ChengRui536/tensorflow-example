# coding=utf-8
import numpy as np
import sklearn.preprocessing as prep #对数据进行预处理的常用模块
import tensorflow as tf

#参数初始化方法-xavier初始化器，会根据某一层网络的输入、输出节点数量自动调整最合适的布局
#标准的均匀分布的xavier初始化器
def xavier_init(fan_in,fan_out,constant=1): #fan_in是输入节点的数量，fan_out是输出节点的数量
    low=-constant * np.sqrt(6.0/(fan_in+fan_out))
    high=constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32) #创建了一个均匀分布


#数据标准化处理，让数据变成均值为0，标准差为1的分布
def standard_scale(X_train,X_test):
    preprocessor=prep.StandardScaler().fit(X_train) #先在训练数据上fit出一个共用的Scaler
    #保证训练数据和测试数据都使用完全相同的Scaler，才能保证后面模型处理数据时的一致性
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train,X_test


#随机获取block数据，数据的起始位置随意，大小为batch_size，不放回抽样，可提高数据的利用效率
def get_random_block_from_data(data,batch_size):
    start_index=np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]