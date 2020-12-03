#!/usr/bin/env python
# coding: utf-8

# In[74]:


import os, glob,pandas as pd, numpy as np
import cv2
from PIL import Image
import shutil





# 구분할 defect의 코드(결함랭크) 
defect_code = 'R10'


# 분리할 이미지가 있는 폴더 
##########################################
img_dir = 'D:/CASTING/' + defect_code + '/new_split_image'
##########################################

# 분리할 이미지를 지정한 엑셀파일 
#################################################################################################
file_path = img_dir + '/' + defect_code + '_이미지구분.xlsx'
#################################################################################################






# 이미지 구분 경로 선언 및 폴더 생성
#############################################
RESULT_SAVE_PATH_0 = img_dir + '/' + defect_code + '_split'
RESULT_SAVE_PATH_defect = RESULT_SAVE_PATH_0 + '/' + defect_code 
RESULT_SAVE_PATH_NO_defect = RESULT_SAVE_PATH_0 + '/NO_' + defect_code
#############################################

def mkdir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

mkdir(RESULT_SAVE_PATH_0)
mkdir(RESULT_SAVE_PATH_defect)
mkdir(RESULT_SAVE_PATH_NO_defect)

# 분리할 이미지 파일 리스트 읽어오기 
filenames = []
files = glob.glob(img_dir + "/*.BMP")
for i, f in enumerate(files):
    filenames.append(f.split("\\")[1].split(".")[0])

# 분리할 이미지를 지정한 엑셀 파일을 읽어오기 

df = pd.read_excel(file_path, encoding='CP949',index_col = '결함번호')

# df = pd.read_csv(file_path, encoding='CP949',index_col = '결함번호')

data = df[['분류','결함랭크(1～10)']]
data = data.fillna('')


for index,row in data.iterrows():
    ## 구분할 이미지
    if row['분류'] != '':
        # img_bw = Image.open(files[index-1])
        mkdir(RESULT_SAVE_PATH_defect + '/defect_origin')
        # img_bw.save(RESULT_SAVE_PATH_defect + '/defect_origin/' + filenames[index-1]+ '.BMP')
        shutil.copy2(files[index-1], RESULT_SAVE_PATH_defect + '/defect_origin/' + filenames[index-1]+ '.BMP' )
    ## 구분할 이미지 외의 이미지
    if row['분류'] == '' : 
        # img_no_bw = Image.open(files[index-1])
        if row['결함랭크(1～10)'] == defect_code : 
            mkdir(RESULT_SAVE_PATH_NO_defect + '/defect_err')
            # img_no_bw.save(RESULT_SAVE_PATH_NO_defect + '/defect_err/' + filenames[index-1]+ '.BMP')
            shutil.copy2(files[index-1], RESULT_SAVE_PATH_NO_defect + '/defect_err/' + filenames[index-1]+ '.BMP' )
        elif row['결함랭크(1～10)'] != 'S1' and row['결함랭크(1～10)'] != 'S2':
            mkdir(RESULT_SAVE_PATH_NO_defect + '/' + row['결함랭크(1～10)'])
            # img_no_bw.save(RESULT_SAVE_PATH_NO_defect + '/' + row['결함랭크(1～10)'] +  '/' + filenames[index-1]+ '.BMP')
            shutil.copy2(files[index-1], RESULT_SAVE_PATH_NO_defect + '/' + row['결함랭크(1～10)'] +  '/' + filenames[index-1]+ '.BMP' )
    os.remove(files[index-1])

