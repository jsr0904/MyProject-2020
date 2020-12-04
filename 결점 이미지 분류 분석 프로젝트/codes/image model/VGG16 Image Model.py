#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:


X = np.array(X)
Y = np.array(y)


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

xy = (X_train, X_test, Y_train, Y_test)

if os.path.exists('C:/cpi_image_test2/numpy_data_gray') is False:
    os.mkdir('C:/cpi_image_test2/numpy_data_gray')
np.save("C:/cpi_image_test2/numpy_data/binary_image_data_gray.npy", xy)


# In[2]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = np.load('C:/cpi_image_test2/numpy_data/binary_image_data_gray.npy',allow_pickle = True)
print(X_train.shape)
print(X_test.shape)
print(np.bincount(y_train))
print(np.bincount(y_test))


# In[3]:


X_train = X_train.reshape(X_train.shape[0], 256, 256, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 256, 256, 1).astype('float32') / 255

# X_train = X_train.astype('float32') / 255
# X_test = X_test.astype('float32') / 255


# In[4]:


X_train.shape


# In[5]:


from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Convolution2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from tensorflow.keras.models import Model


# In[7]:


input_tensor = Input(shape=(256, 256, 1), dtype='float32', name='input')
 
x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(input_tensor)
x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = MaxPooling2D((2,2))(x)
 
x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = MaxPooling2D((2,2))(x)
 
x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = MaxPooling2D((2,2))(x)
 
x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = MaxPooling2D((2,2))(x)
 
x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = MaxPooling2D((2,2))(x)
 
x = Flatten()(x)
x = Dense(4096, kernel_initializer='he_normal')(x)
x = Dense(2048, kernel_initializer='he_normal')(x)
x = Dense(1024, kernel_initializer='he_normal')(x)
output_tensor = Dense(1, activation='sigmoid')(x)
 
myvgg = Model(input_tensor, output_tensor)
myvgg.summary()


# In[10]:


from tensorflow.keras.optimizers import Adam,RMSprop,SGD
myvgg.compile(optimizer=RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[11]:


# checkpoint
checkpoint_path = 'C:/cpi_image_test2/model/checkpoint_VGG16.ckpt'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

patient = 5
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
            monitor='val_accuracy', 
            patience=patient
        )]


# In[12]:


myvgg.fit(X_train, y_train, batch_size=64, epochs=40, validation_split=0.15,  callbacks= callback_list)

myvgg.load_weights(checkpoint_path)


# In[28]:


print("정확도 : %.2f " %(myvgg.evaluate(X_test, y_test)[1]))


# In[29]:


model.save("C:/cpi_image_test2/model/VGG16.h5")


# In[ ]:




