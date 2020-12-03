#!/usr/bin/env python
# coding: utf-8


import os
import time
import pyhdb
import numpy as np
import pandas as pd
import pandas.io.sql as pd_sql
import pytz
import statsmodels.api as sm
import xgboost as xgb
import warnings
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
from influxdb import InfluxDBClient,DataFrameClient 
from copy import deepcopy
from sklearn.model_selection import KFold, GridSearchCV
import pickle
import logging,datetime
import pymssql


############################################### 파이선 로깅 ###########################################
## 1. 로거 생성
mylogger = logging.getLogger("my") 
mylogger.setLevel(logging.INFO) # 로깅레벨(DEBUG, INFO, WARNING, ERROR, CRITICAL) 중 INFO 레벨로 설정

## 2. 핸들러 설정 -> 로깅 정보가 console에 출력하도록 설정    
stream_hander = logging.StreamHandler()
mylogger.addHandler(stream_hander)

## 3. 파일 핸들러 설정 -> 로깅 정보가 파일로도 동시에 출력
file_handler = logging.FileHandler('E:/cms_modeling_log/modeling.log') ## 파일 위치 지정
mylogger.addHandler(file_handler)

current = datetime.datetime.now()
current_date = current.strftime('%Y-%m-%d %H:%M:%S')

mylogger.info("모델링 시작 시간 = " + str(current_date))


############################################### DB에 저장된 설명변수 X, 타겟변수 Y 정보 가져오기 ###########################################
connection = pymssql.connect(host='',
                       port= '',
                       user ='',
                       password ='',
                       database=''
)

query = "select B.EQP_NO, C.NAME, A.X_VAR, A.Y_VAR, A.X_RUN_COND, A.Y_RUN_COND from CMS_MODELING A, EQUIPMENT B, PROCESS_LIST C where A.EQP_ID = B.EQP_ID and B.PRCS_ID = C.PRCS_ID"
db_value = pd.read_sql(query,con=connection)

db_value.columns = ['설비','공정','X','Y','운전조건_X','운전조건_Y']

get = pd.DataFrame()
get = pd.concat([db_value['Y'], db_value['운전조건_Y'].drop_duplicates()])


str1 = 'mean('
str2 = ') AS '
count = 0
sql_first = 'SELECT '
sql_last = ' FROM "EDGE_TAG_HISTORY_TS" WHERE time > now() - 365d GROUP BY time(1m)'

for index, value in get.items():
    count = count + 1
    if count == len(get): 
        str_store = str1 + '"' + str(value) + '"' + str2 + '"' + str(value) + '"' 
        sql_first = sql_first + str_store
    else:
        str_store = str1 + '"' + str(value) + '"' +  str2 + '"' + str(value) + '",' 
        sql_first = sql_first + str_store



############################################### 실시간 공정 데이터 LOAD ###########################################
start = time.time()
#### cms influx db 불러오는 함수  ####
#### database 이름, host, port, user 및 password를 파라미터로 받아오면 됨

def get_ifdb_cms(db='', host='', port=, user='', passwd=''):
    client = DataFrameClient(host, port, user, passwd, db)
    return client

ifdb = get_ifdb_cms() ## influx db 호출

#### influx data 불러오기 및 DataFrame 형식으로 변환 ####
result = ifdb.query(sql_first + sql_last)
column = next(iter(result))
output_table   = result[column]
data = output_table 
data.index.name = 'DATE' ## index 이름을 DATE로 변경

mylogger.info("데이터 LOAD 시간 :" + str(time.time() - start))


############################################### 결측치 처리 ###########################################

start = time.time()

data = data.fillna(method='pad')
data = data.fillna(method='backfill')
data_fill_row = data.dropna(axis=0)

mylogger.info("빈값 처리 시간 :" + str(time.time() - start))



############################################### 모델 저장 위치 지정 ###########################################
def mkdir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

current = datetime.datetime.now() ## 현재 시각 가져오기
current_date = current.strftime('%Y%m%d') ## 현재 시각은 YYYYMMDD 형태로 변경
current_date_model_path = 'C:/' + current_date ## 모델 저장 폴더 경로 설정

mkdir(current_date_model_path) ## 모델 저장 폴더 생성

os.chdir(current_date_model_path + '/')
os.getcwd()


############################################### XGBOOST Regression 을 이용한 모델링 ###########################################
for sulbi_name in db_value['설비'].drop_duplicates():
    
    excel_value = pd.DataFrame()
    sulbi_condition = db_value['설비'] == sulbi_name
    excel_value = db_value[sulbi_condition]
    excel_value = excel_value.reset_index().drop('index', axis=1)
    
    X_name = excel_value.X.apply(lambda x : str(x).split(","))
    Y_name = excel_value.Y.apply(lambda x : str(x))
    Fac_name = excel_value.설비.apply(lambda x : str(x))
    Fac_no = excel_value.공정.apply(lambda x : str(x))
    X_drive = excel_value.운전조건_X.apply(lambda x : str(x).split(","))
    Y_drive = excel_value.운전조건_Y.apply(lambda x : str(x))

    dummy_file_name = pd.DataFrame(columns=('File Name','NewX'))

    for count in range(len(X_name)):
        start = time.time()
        ###################### X 설명변수 설정 ############################
        X_all = data_fill_row[X_name[count]]
        ###################### X 설명변수 Drive ##########################
        Drive_X = data_fill_row[X_drive[count]]
        drive_x = Drive_X[Drive_X == 1].dropna(thresh = 1)
        ######## X설명변수와 X Drive join #################################
        x_join = pd.merge(drive_x,X_all,how='left',on='DATE')
        ##################### Y 타켓변수 설정 #############################
        y1_all = pd.to_numeric(data_fill_row[Y_name[count]])
        ##################### Y 타겟변수 Drive ############################
        if Y_drive[count] == '': ## 만일 Drive조건이 없다면
            y_join = y1_all.to_frame()
        else : ## 만일 Drive 조건이 있다면 
            Drive_Y = pd.to_numeric(data_fill_row[Y_drive[count]])
            drive_y = Drive_Y[(Drive_Y.values == 1.0)]
            y_join =  pd.concat([drive_y,y1_all],axis = 1,join='inner')
        ######### 운전조건을 만족하는 X와 운전조건을 만족하는 Y가 다르니 Y를 기준으로 left join #########
        xy = pd.merge(y_join,x_join,how = 'left',on='DATE')
        mylogger.info("운전 조건")
        ###########################  Binning ##################################
        xy['binning'] = pd.qcut(xy[Y_name[count]], q=4)
        xy['binning'] = xy.binning.astype(str)
        ########################### 현재와 일치하는 binning 구간의 데이터만 가져오기.
        xy_binning = pd.DataFrame()
        value = xy.tail(1).loc[:,'binning']
        filterling = (xy.binning == value[0])
        xy_binning = xy.loc[filterling,:]
        xy_binning = xy_binning.drop(['binning'], axis=1) 
        ############################ Outlier 제거 ############################
        temp = xy_binning.copy()
        def remove_outlier(df_in, col_name):
            q1 = df_in[col_name].quantile(0.25)
            q3 = df_in[col_name].quantile(0.75)
            iqr = q3-q1 #Interquartile range
            fence_low  = q1-15*iqr
            fence_high = q3+15*iqr
            df_out = df_in[~((df_in < (fence_low)) |(df_in > (fence_high))).any(axis=1)]
            return df_out
        df_outlier_row = remove_outlier(temp, temp.columns)
        mylogger.info("아웃라이어 제거")
        ###################### X 설명변수 설정 ##################################
        X = df_outlier_row[X_name[count]].fillna(method='pad').fillna(method='backfill')
        ##################### Y 타켓변수 설정  ##################################
        y1 = pd.to_numeric(df_outlier_row[Y_name[count]])
        ##################### 변수 선택을 위한 다중 선형 회귀 -> p value<0.05 이하인거 뽑기.
        x_data1 = sm.add_constant(X, has_constant='add')
        multi_model = sm.OLS(y1,x_data1)
        fitted_multi_model = multi_model.fit()
        LRresult = fitted_multi_model.summary2().tables[1]
        newlist = list(LRresult[LRresult['P>|t|']<=0.05].index)[1:]
        ####################### 변수가 안뽑힌 경우 ###############################
        if len(newlist) == 0 :
            mylogger.info("뽑힌변수 없음")
            X1 = X
        ###################### X변수 재설정 ######################################
        else :
            for a in newlist:
                if a == 'const':
                    newlist.remove('const')
            mylogger.info("다중선형회귀 후 변수 뽑기")
            mylogger.info(str(newlist))
            X1 = X[newlist]
        ##################### train and test 나누기, 학습데이터와 평가데이터의 비율을 8:2 
        train_x, test_x, train_y, test_y = train_test_split(X1, y1, test_size = 0.2, random_state = 156)
        #################### 모델 선정  ###############################################  
        model=xgb.XGBRegressor()
        param_grid={'nthread':[4],
                  'objective':['reg:linear'],
                  'booster' :['gbtree'],
                  'learning_rate': [.3], 
                  'max_depth': [6],
                  'min_child_weight': [1],
                  'silent': [1],
                  'subsample': [1],
                  'colsample_bytree': [1]}
        cv = KFold(n_splits=10, random_state=42, shuffle=True)
        tuned_model = GridSearchCV(model, param_grid=param_grid,scoring='neg_mean_squared_error',cv=cv, n_jobs=8)
        tuned_model.fit(train_x,train_y, verbose=1)
        model= tuned_model.best_estimator_
        evals = [(test_x,test_y)]
        xgb_model= model.fit(train_x, train_y, early_stopping_rounds=100,eval_set= evals, verbose=0)
        #################### 모델 저장 #################################################
        file_name = str(Fac_name[count]) +'_'+ str(Fac_no[count])+'_'+ str(Y_name[count]) +'.bin'
        dummy_file_name.loc[count,'File Name'] = file_name
        
        if len(newlist) == 0 :
            dummy_file_name.loc[count,'NewX'] = ','.join(X_name[count])
        else :
            dummy_file_name.loc[count,'NewX'] = ','.join(newlist)
        
        dummy_file_name.loc[count,'Y'] = str(Y_name[count])
        
        pickle.dump(xgb_model,open(dummy_file_name['File Name'][count],"wb"))
        mylogger.info("모델 저장 시간: " + str(time.time() - start))
        #################### 모델 평가 ###############################################
        xgb_model_pred = model.predict(test_x)
        from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
        mse = mean_squared_error(test_y,xgb_model_pred)
        rmse = np.sqrt(mse)

        print(Y_name[count])
        mylogger.info("태그명: " + Y_name[count])
        mylogger.info("=====================================================")
        print('mse = ' , mse)
        print('rmse =', rmse)
        print('Variance score : {0:.3f}'.format(r2_score(test_y,xgb_model_pred)))
    ##################################################################################################################
    path_folder = os.getcwd()+'/bin'
    mkdir(path_folder)
    path = os.getcwd()+'/bin/'+'bin_'+ sulbi_name +'.xlsx'
    writer = pd.ExcelWriter(path)
    dummy_file_name.to_excel(writer)
    writer.save()
    mylogger.info("파일생성")
    mylogger.info("===============================한 설비 끝=================================")




