#!/usr/bin/env python
# coding: utf-8

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
import matplotlib.pyplot as plt
import tensorflow

img_dir = 'C:/cpi_image_test2'
categories = ['BW_image', 'no_BW_image']
np_classes = len(categories)

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


X = np.array(X)
Y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

xy = (X_train, X_test, Y_train, Y_test)

if os.path.exists('C:/cpi_image_test2/numpy_data_gray') is False:
    os.mkdir('C:/cpi_image_test2/numpy_data_gray')
np.save("C:/cpi_image_test2/numpy_data/binary_image_data_gray.npy", xy)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = np.load('C:/cpi_image_test2/numpy_data/binary_image_data_gray.npy',allow_pickle = True)
print(X_train.shape)
print(X_test.shape)
print(np.bincount(y_train))
print(np.bincount(y_test))

X_train = X_train.reshape(X_train.shape[0], 256, 256, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 256, 256, 1).astype('float32')

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

from tensorflow.keras.layers import Input,Concatenate,concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization,Activation 
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import GlobalAveragePooling2D,ZeroPadding2D,Add 
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

# create model
input_img = Input(shape=(256, 256, 1), name='main_input')
 
#VGG Net
x1 = Conv2D(64, (3, 3))(input_img)
x1 = Activation('relu')(x1)
x1 = Conv2D(64, (3, 3))(x1)
x1 = Activation('relu')(x1)
x1 = MaxPooling2D()(x1)
x1 = Conv2D(64, (3, 3))(x1)
x1 = Activation('relu')(x1)
x1 = Conv2D(64, (3, 3))(x1)
x1 = Activation('relu')(x1)
x1 = MaxPooling2D()(x1)
x1 = Conv2D(64, (3, 3))(x1)
x1 = Activation('relu')(x1)
x1 = MaxPooling2D()(x1)
x1 = Flatten()(x1)
x1 = Dense(256)(x1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = Dense(256)(x1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
 
#Res Net
x = Conv2D(64, (3, 3))(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = (ZeroPadding2D((1,1)))(x)
x = Conv2D(64, (3, 3))(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(1, (3, 3))(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = (ZeroPadding2D((1,1)))(x)
#x = merge([x, input_img], mode='sum')
x = Concatenate(axis=1)([x, input_img])
x = Flatten()(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
 
x = concatenate([x1, x])
out = Dense(1, activation='sigmoid')(x)
 
# Compile model
model = Model(inputs=input_img, outputs=out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model_dir = 'C:/cpi_image_test2/model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = model_dir + "/binary_classify_ResNet_VGG.model"
if not os.path.exists(model_path):
    os.mkdir(model_path)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

patient = 3
callback_list = [
        ReduceLROnPlateau(
            moniter = 'val_loss',
            factor = 0.1,
            patience = patient,
            min_lr = 0.00001,
            mode = 'auto'
        ),
        ModelCheckpoint(
            filepath=model_path, 
            monitor='val_accuracy', 
            verbose=1,
            mode = 'max',
            save_best_only=True
        ),
        EarlyStopping(
            monitor='val_accuracy', 
            patience=patient
        )]

history = model.fit(X_train, y_train, batch_size=64, epochs=32, validation_split=0.15,  callbacks= callback_list)

print("정확도 : %.2f " %(model.evaluate(X_test, y_test)[1]))
