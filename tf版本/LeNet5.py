import tensorflow as tf

#输入
image_size=28 #默认方形
n_channels=1 #输入图片的channel 

#网络参数
n_classes=10

# 第一卷积层尺寸和深度
CONV1_DEEP=32
CONV1_SIZE=5
#第二卷积层的尺寸和深度
CONV2_DEEP=64
CONV2_SIZE=5
#全连接层的节点个数
FC_SIZE=512

def LeNet5(input_tensor,trainFlag, regularizer):
    with tf.variable_scope('conv1'):
        conv1_weights=tf.get_variable(
            'weights',[CONV1_SIZE,CONV1_SIZE,n_channels,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        
    pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.variable_scope('conv2'):
        conv2_weights=tf.get_variable(
            'weights', [CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable('bias',[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2=tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    pool2=tf.nn.max_pool(relu2, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    # 对LeNet-5，第四层的输出为7x7x64的矩阵
    # 全连接层需要将这个矩阵拉直成一个向量
    # pool2.gets_shape函数得到第四层的输出，维数为batch_size*7*7*64
    pool_shape=pool2.get_shape().as_list()
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3];
    reshaped=tf.reshape(pool2, [-1,nodes])
    with tf.variable_scope('fc1'):
        fc1_weights=tf.get_variable(
            'weights',[nodes, FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer !=None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases=tf.get_variable('bias',[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if trainFlag: fc1=tf.nn.dropout(fc1, DROPOUT)
    with tf.variable_scope('fc2'):
        fc2_weights=tf.get_variable('weights', [FC_SIZE, n_classes],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer !=None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases=tf.get_variable('bias',[n_classes],initializer=tf.constant_initializer(0.1))
        fc2=tf.matmul(fc1,fc2_weights)+fc2_biases
    return fc2
