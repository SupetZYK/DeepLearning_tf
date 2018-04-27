from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D
from keras.optimizers import SGD
from keras.utils import to_categorical

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

def Lenet5():
    #两层卷积
    model=Sequential()
    model.add(Conv2D(filters=CONV1_DEEP, kernel_size=(CONV1_SIZE,CONV1_SIZE), padding='valid', input_shape=(image_size,image_size,n_channels), activation='tanh'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=CONV2_DEEP, kernel_size=(CONV2_SIZE,CONV2_SIZE), padding='valid', activation='tanh'))
    model.add(MaxPool2D(pool_size=(2,2)))
    #全链接
    model.add(Flatten())
    model.add(Dense(FC_SIZE, activation='tanh'))
    model.add(Dense(n_classes, activation='softmax'))
    #训练相关
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy']) 
    return model