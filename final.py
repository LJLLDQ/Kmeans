
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams['font.size'] = 8
plt.rcParams['figure.figsize'] = (8,8)

import os
import numpy as np
import cv2
import pickle

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K

from sklearn.cluster import KMeans



# 导入图片

img_width, img_height, channels = 128, 128, 3
input_shape = (img_width, img_height, channels)

def load_data():
    dir = 'path' # 路径
    files = ["%s/%s" % (dir, x) for x in os.listdir(dir)]
    arr = np.empty((len(files), img_width, img_height, channels), dtype=np.float32)
    for i, imgfile in enumerate(files):
        img = load_img(imgfile)
        x = img_to_array(img).reshape(img_width, img_height, channels)
        x = x.astype('float32') / 255.
        arr[i] = x
    return arr

X = load_data()

print(X.shape)


# 搭建 autoencoder 模型

model = Sequential()

# encoder 部分: ( conv + relu + maxpooling ) * 3
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# decoder 部分: ( conv + relu + upsampling ) * 3 与 encoder 过程相反
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same'))

model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.summary()

model.fit(X, X, epochs=20, batch_size=128, shuffle=True)


# 保存模型, 方便下次调用

model.save('taptap-logo-autoencoder-model.h5')


# 取 100 个样本来做下分析

X_sample = X[:100]
print(X_sample.shape


# 所有图片 encoded 之后的数据

X_encoded = np.empty((len(X), 16, 16, 8), dtype='float32')

step = 100
for i in range(0, len(X), step):
    x_batch = get_encoded([X[i:i+step]])[0]
    X_encoded[i:i+step] = x_batch

print(X_encoded.shape)


# reshape, 其实相当于 flatten, 之后给 KMeans 用

X_encoded_reshape = X_encoded.reshape(X_encoded.shape[0], X_encoded.shape[1]*X_encoded.shape[2]*X_encoded.shape[3])
print(X_encoded_reshape.shape)


# KMeans 聚类

n_clusters = 100 # 分几类

km = KMeans(n_clusters=n_clusters)
km.fit(X_encoded_reshape)


plt.figure(figsize=(20, 20))

cluster = 1 # 看第几类的聚类
rows, cols = 1, 1 # 
start = 0

labels = np.where(km.labels_==cluster)[0][start:start+rows*cols]
for i, label in enumerate(labels):
    plt.subplot(rows, cols, i+1)
    plt.imshow(X[label])
    plt.axis('off')