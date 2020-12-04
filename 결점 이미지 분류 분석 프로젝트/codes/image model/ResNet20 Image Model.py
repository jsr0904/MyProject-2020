#!/usr/bin/env python
# coding: utf-8

# In[64]:


import cv2
import os
import glob
import shutil
import random
import string
import numpy as np
import pandas as pd
from PIL import Image
import os, glob, numpy as np
from keras_preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt

img_dir = 'C:/cpi_image_test2'
categories = ['BW_image', 'no_BW_image']
np_classes = len(categories)


# In[15]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
data_aug_gen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.5,
    zoom_range=[0.8, 2.0],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
    )

img_dir = 'C:/cpi_image_test2'
img_dir_detail = img_dir + "/" + "BW_image" + "/"
files = glob.glob(img_dir_detail + '*.png')
save_to_dir = 'C:/cpi_image_test2/BW_image'


# In[ ]:


## DATA Argumentation 
for i, f in enumerate(files):
    png_name = ''.join(f.split()).split('\\')[1][:-4]
    save_to_dir1 = save_to_dir + '/' + png_name
    if not os.path.exists(save_to_dir1):
        os.mkdir(save_to_dir1)
    img = load_img(f)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir=save_to_dir1, save_prefix='plus_', save_format='png'):
        i += 1
        if i > 10:
            break


# In[16]:


image_w = 256
image_h = 256

X = []
y = []

for idx, BW in enumerate(categories):

    img_dir_detail_1 = img_dir + "/" + BW + "/"

    if idx == 0 :           #idx==0 -> BW_image
        files = glob.glob(img_dir_detail_1 + '**/*.png') + glob.glob(img_dir_detail_1 + '*.png') 
    else:                   #idx==1 -> no_BW_image
        files = glob.glob(img_dir_detail_1 + '**/*.png')
    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert("L")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            X.append(data)
            y.append(idx)
        except:
            print(BW, str(i)+" 번째에서 에러 ")


# In[17]:


X = np.array(X)
Y = np.array(y)


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

xy = (X_train, X_test, Y_train, Y_test)

if os.path.exists('C:/cpi_image_test2/numpy_data_gray') is False:
    os.mkdir('C:/cpi_image_test2/numpy_data_gray')
np.save("C:/cpi_image_test2/numpy_data/binary_image_data_gray.npy", xy)


# In[90]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = np.load('C:/cpi_image_test2/numpy_data/binary_image_data_gray.npy',allow_pickle = True)
print(X_train.shape)
print(X_test.shape)
print(np.bincount(y_train))
print(np.bincount(y_test))


# In[91]:


X_train = X_train.reshape(X_train.shape[0], 256, 256, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 256, 256, 1).astype('float32') / 255

# X_train = X_train.astype('float32') / 255
# X_test = X_test.astype('float32') / 255


# In[92]:


X_train.shape


# In[93]:


from tensorflow.keras.layers import Input,Concatenate,concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization,Activation 
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import GlobalAveragePooling2D,ZeroPadding2D,Add,AveragePooling2D,add
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# In[94]:


def get_resnet():
    # In order to make things less confusing, all layers have been declared first, and then used
    
    # declaration of layers
    input_img = Input((256,256,1), name='input_layer')
    zeroPad1 = ZeroPadding2D((1,1), name='zeroPad1')
    zeroPad1_2 = ZeroPadding2D((1,1), name='zeroPad1_2')
    layer1 = Conv2D(6, (3, 3), strides=(2, 2), kernel_initializer='he_uniform', name='major_conv')
    layer1_2 = Conv2D(16, (3, 3), strides=(2, 2), kernel_initializer='he_uniform', name='major_conv2')
    zeroPad2 = ZeroPadding2D((1,1), name='zeroPad2')
    zeroPad2_2 = ZeroPadding2D((1,1), name='zeroPad2_2')
    layer2 = Conv2D(6, (3, 3), strides=(1,1), kernel_initializer='he_uniform', name='l1_conv')
    layer2_2 = Conv2D(16,(3, 3), strides=(1,1), kernel_initializer='he_uniform', name='l1_conv2')

    zeroPad3 = ZeroPadding2D((1,1), name='zeroPad3')
    zeroPad3_2 = ZeroPadding2D((1,1), name='zeroPad3_2')
    layer3 = Conv2D(6, (3, 3), strides=(1, 1), kernel_initializer='he_uniform', name='l2_conv')
    layer3_2 = Conv2D(16,(3, 3), strides=(1, 1), kernel_initializer='he_uniform', name='l2_conv2')

    layer4 = Dense(64, activation='relu', kernel_initializer='he_uniform', name='dense1')
    layer5 = Dense(16, activation='relu', kernel_initializer='he_uniform', name='dense2')

    final = Dense(1, activation='sigmoid', kernel_initializer='he_uniform', name='classifier')
    
    # declaration completed
    
    first = zeroPad1(input_img)
    second = layer1(first)
    second = BatchNormalization(name='major_bn')(second)
    second = Activation('relu', name='major_act')(second)

    third = zeroPad2(second)
    third = layer2(third)
    third = BatchNormalization(name='l1_bn')(third)
    third = Activation('relu', name='l1_act')(third)

    third = zeroPad3(third)
    third = layer3(third)
    third = BatchNormalization(name='l1_bn2')(third)
    third = Activation('relu', name='l1_act2')(third)

    res = add([third, second])
    #res = merge([third, second], mode='sum', name='res')
    first2 = zeroPad1_2(res)
    second2 = layer1_2(first2)
    second2 = BatchNormalization(name='major_bn2')(second2)
    second2 = Activation('relu', name='major_act2')(second2)


    third2 = zeroPad2_2(second2)
    third2 = layer2_2(third2)
    third2 = BatchNormalization(name='l2_bn')(third2)
    third2 = Activation('relu', name='l2_act')(third2)

    third2 = zeroPad3_2(third2)
    third2 = layer3_2(third2)
    third2 = BatchNormalization(name='l2_bn2')(third2)
    third2 = Activation('relu', name='l2_act2')(third2)

    res2 = add([third2, second2])
    #res2 = merge([third2, second2], mode='sum', name='res2')

    res2 = Flatten()(res2)

    res2 = layer4(res2)
    res2 = Dropout(0.4, name='dropout1')(res2)
    res2 = layer5(res2)
    res2 = Dropout(0.4, name='dropout2')(res2)
    res2 = final(res2)
    model = Model(inputs= input_img, outputs=res2)

    return model


# In[95]:


res = get_resnet()

from tensorflow.keras.optimizers import Adam,RMSprop,SGD
optimizer = SGD(lr=0.01,decay = 5e-4, momentum=0.9)

res.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])

res.summary()


# In[100]:


# checkpoint
checkpoint_path = 'C:/cpi_image_test2/model/checkpoint_resnet20_ver1.ckpt'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

patient = 4
callback_list = [
        ReduceLROnPlateau(
            moniter = 'val_loss',
            factor = 0.1,
            patience = patient,
            min_lr = 0.00001,
            mode = 'auto'
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss', 
            verbose=1,
            mode = 'auto',
            save_weights_only=True,
            save_best_only=True
        ),
        EarlyStopping(
            monitor='val_acc', 
            patience=patient
        )]


# In[101]:


res.fit(X_train, y_train, batch_size=64, epochs=20, validation_split=0.15,  callbacks= callback_list)

res.load_weights(checkpoint_path)


# In[102]:


print("정확도 : %.2f " %(res.evaluate(X_test, y_test)[1]))


# In[103]:


res.save("C:/cpi_image_test2/model/resnet20_ver1.h5")


# In[ ]:




