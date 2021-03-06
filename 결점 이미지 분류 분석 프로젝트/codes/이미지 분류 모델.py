# -*- coding: utf-8 -*-

# get_ipython().system('pip install Image')
# get_ipython().system('pip install numpy')
# get_ipython().system('pip install sklearn')

# get_ipython().system('pip install opencv-python')
# get_ipython().system('pip install keras==2.1.6')
# get_ipython().system('pip install tensorflow==2')
# get_ipython().system('pip install matplotlib')
# get_ipython().system('pip install openpyxl')
# get_ipython().system('pip install xlrd')


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
import os, glob, sys, numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import cv2
import shutil
from tensorflow.keras.models import load_model
from datetime import datetime



# input_set_shift : 위치이동 비율
input_set_shift = 0.3
# input_set_zoom : 확대 비율 
input_set_zoom = 2.0


# 결점 name(공백없는 영문표기)
input_set_defect = "R10"


# img_dir : 입력 이미지 상위 경로 
img_dir = 'D:/CASTING/' + input_set_defect + '/analysis_split_target_image_result'

# 모델 버전 
test_ver = '08a'

# 모델 경로 설정 
img_dir_model = 'D:/CPI_model/CASTING_model/'+ input_set_defect
categories = ['defect', 'no_defect']
np_classes = len(categories)

# 이미지 사이즈
# image_w = 256
# image_h = 256
image_w = 64
image_h = 64

size_str = str(image_w)

model_name = 'CASTING_v1'

# data_aug_gen : 이미지 확대 및 위치이동 함수
data_aug_gen = ImageDataGenerator(rescale=1. / 255,
    rotation_range=0,
    width_shift_range = input_set_shift,
    height_shift_range = input_set_shift,
    shear_range=0.5,
    zoom_range=[1.0, input_set_zoom],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

# img_dir_detail : 원본을 읽어올 경로
img_dir_detail = img_dir + "/result/defect" + "/"
files = glob.glob(img_dir_detail + '*.BMP')
# save_to_dir : generate 이미지 파일을 져장할 경로
save_to_dir = img_dir + '/result/defect'

## DATA Argumentation 
for i, f in enumerate(files):
    png_name = ''.join(f.split()).split('\\')[1][:-4]
    # save_to_dir1 = save_to_dir + '/' + png_name
    save_to_dir1 = save_to_dir 
    if not os.path.exists(save_to_dir1):
        os.mkdir(save_to_dir1)
    img = load_img(f)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir=save_to_dir1, save_prefix='plus_', save_format='BMP'):
        i += 1
        if i > 29:
            break

pixel = image_h * image_w * 3
X = []
y = []

# 학습 대상 데이터 읽어오기 
for idx, BW in enumerate(categories):
    img_dir_detail = img_dir + "/result/" + BW + "/"

    if idx == 0 :           #idx==0 -> 검출 대상 이미지
        files = glob.glob(img_dir_detail + '*.BMP')
    else:                   #idx==1 -> 검출 대상 이외 이미지
        files = glob.glob(img_dir_detail + '*.BMP')

    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            X.append(data)
            y.append(idx)   #Y는 0 아니면 1이니까 idx값으로 넣음
            if i % 300 == 0:
                print(BW, " : ", f)
        except:
            print(BW, str(i)+" 번째에서 에러 ")
X = np.array(X)
Y = np.array(y)

# train test 데이터 나누기 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

xy = (X_train, X_test, Y_train, Y_test)



# 모델 저장 경로 생성 
model_path = img_dir_model +'/' + model_name
# 모델 load 
model = load_model( model_path +'/' + model_name + '.h5' )

# 모델에 이미지 데이터 재학습 (for에서 추가 2회 : 총 2회) 
for model_step in range(2, 4) :
    print(model_step)
    
    tmp_files1 = glob.glob(model_path + '/*.*')
    tmp_files2 = glob.glob(model_path + '/**/*.*')
    
    model_step_str = str(model_step)
    
    model_path_back = img_dir_model + '/' + model_name + '_' + datetime.today().strftime("%Y%m%d%H%M%S")
    if not os.path.exists(model_path_back):
        os.mkdir(model_path_back)
    if not os.path.exists(model_path_back + '/assets'):
        os.mkdir(model_path_back + '/assets')
    if not os.path.exists(model_path_back + '/variables'):
        os.mkdir(model_path_back + '/variables')        
    if os.path.exists(model_path_back + '/numpy_data' + test_ver) is False:
        os.mkdir(model_path_back + '/numpy_data' + test_ver)
    # np.save(model_path + "/numpy_data" + test_ver + "/" + model_name + ".npy", xy)
    
    for i, f in enumerate(tmp_files1):
        print(i, f)
        shutil.copy2(f, model_path_back +'/' + f.split("\\")[1] )
        
    for i, f in enumerate(tmp_files2):
        print(i, f)
        shutil.copy2(f, model_path_back +'/' + f.split("\\")[1] +'/' + f.split("\\")[2] )
        
    if model_step == 2 :
# 추가 학습 데이터 저장 
        if os.path.exists(model_path + '/numpy_data' + test_ver) is False:
            os.mkdir(model_path + '/numpy_data' + test_ver)
        # np.save(model_path + "/numpy_data" + test_ver + "/" + model_name + ".npy", xy)
        
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255
        X = X.astype('float32') / 255      
    
    checkpoint = ModelCheckpoint(filepath= model_path , monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15)
    
    history = model.fit(X, Y, batch_size=64, epochs=1000, validation_data=(X_test, Y_test), callbacks=[checkpoint, early_stopping])
    
    model.save( model_path +'/' +  model_name + '.h5' )

    model = load_model( model_path +'/' +  model_name + '.h5' )


# 학습 대상 데이터 삭제

for idx, BW in enumerate(categories):
    try:
        # if idx == 0 :           
            img_dir_detail = img_dir + "/result/" + BW + "/"
            shutil.rmtree(img_dir_detail)
            os.mkdir(img_dir_detail)
    except:
        print("삭제할 경로가 없습니다")
