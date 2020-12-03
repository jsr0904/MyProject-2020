#!/usr/bin/env python
# coding: utf-8

# In[1]:


import schedule 
import time
import datetime
from influxdb import InfluxDBClient,DataFrameClient 
from pandas import DataFrame
import os
import pandas as pd
import pickle
import warnings
import pymssql

#os.chdir("/0714/")
connection = pymssql.connect(host='172.17.48.20',
                       port= '5008',
                       user ='cms_admin', 
                       password ='adminCms@508!', 
                       database='HISTORIAN'
)

query = "select B.EQP_NO, C.NAME, A.X_VAR, A.Y_VAR, A.X_RUN_COND, A.Y_RUN_COND from CMS_MODELING A, EQUIPMENT B, PROCESS_LIST C where A.EQP_ID = B.EQP_ID and B.PRCS_ID = C.PRCS_ID"
db_value = pd.read_sql(query,con=connection)

db_value.columns = ['설비','공정','X','Y','운전조건_X','운전조건_Y']

get = pd.DataFrame()
get = db_value['Y'].drop_duplicates()

str1 = 'mean('
str2 = ') AS '
count1 = 0
sql_first = 'SELECT '
sql_last = ' FROM "EDGE_TAG_HISTORY_TS" order by time desc limit 5'

for index, value in get.items():
    count1 = count1 + 1
    if count1 == len(get): 
        str_store = '"' + str(value) + '"'
        sql_first = sql_first + str_store
    else:
        str_store = '"' + str(value) + '",' 
        sql_first = sql_first + str_store

sql = sql_first + sql_last
sulbi_names = db_value['설비'].drop_duplicates()


# In[3]:


current = datetime.datetime.now()
current_date = current.strftime('%Y-%m-%d %H:%M:%S')
print("Current Time = " , current_date)
yesterday = current - datetime.timedelta(days = 1)
model_check = yesterday.strftime('%Y%m%d')


# In[4]:


model_check


# In[ ]:





# In[2]:


def job():
    ###################모델 변경#################################
    current = datetime.datetime.now()
    current_date = current.strftime('%Y-%m-%d %H:%M:%S')
    print("Current Time = " , current_date)
    yesterday = current - datetime.timedelta(days = 1)
    model_check = yesterday.strftime('%Y%m%d')
    
    folders = os.listdir('C:/')
    for folder_name in folders:
        if model_check == folder_name:
            os.chdir('C:/' + folder_name)
            print("모델 경로 바뀜: "+ folder_name)
            break
    ##################################################
    start = time.time()
    ##################################### influx DB 테스트용 ###############################
    def get_ifdb_cms(db, host='172.17.48.31', port=8086, user='python', passwd='python123'):
        client = DataFrameClient(host, port, user, passwd, db)
        return client

    ifdb = get_ifdb_cms(db='pis')
    result = ifdb.query(sql)
    column = next(iter(result))
    output_table   = result[column]
    data = output_table
    ##################################### 예측 #############################################
    dummy = pd.DataFrame()
    dummy_pred = pd.DataFrame()

    for sulbi_name in sulbi_names:
        print(sulbi_name)
        
        file_path1 = os.getcwd()+'/bin/'+'bin_' + sulbi_name +'.xlsx'
        dummy_model_name = pd.read_excel(file_path1)

        X_new_name = dummy_model_name['NewX'].apply(lambda x : str(x).split(","))
        Y_name = dummy_model_name['Y'].apply(lambda x : str(x))
        
        for count in range(len(X_new_name)):
            try:
                test_x = data[X_new_name[count]]
            except KeyError:
                ## PRE 이전값 가져오는 경우
                pre_data = ifdb.query('SELECT' + ' '+'PRE_' + Y_name[count] + ' '+ 'FROM "EDGE_TAG_HISTORY_TS" order by time desc limit 1')
                column1 = next(iter(pre_data))
                output_table1 = pre_data[column1]
                data1 = output_table1
                dummy = pd.DataFrame(index = data.index).sort_index()
                dummy['진동센서'] = 'PRE_'+ Y_name[count]
                dummy['Predict'] = data1.iloc[0,0]
                dummy_pred = pd.concat([dummy_pred,dummy],axis=0)
                continue
            ##################### model 불러오기 및 예측
            loaded_model = pickle.load(open(dummy_model_name['File Name'][count], "rb"))
            print(dummy_model_name['File Name'][count])
            xgb_model_pred = loaded_model.predict(test_x)

            ## 예측값 과 실제값 합침(DataFrame 만들어서)
            dummy = pd.DataFrame(xgb_model_pred,columns=['Predict'],index = test_x.index).sort_index()
            print(Y_name[count])
            dummy['진동센서'] = 'PRE_'+ Y_name[count]

            dummy_pred = pd.concat([dummy_pred,dummy],axis=0)

    gogo_pv = pd.pivot_table(dummy_pred,index = dummy_pred.index,columns='진동센서',values='Predict')
    ##### 이력 ######
    for index, row in gogo_pv.iterrows():
        if index.second % 5 == 0 : gogo_pv.loc[index,'t5s'] = "1"
        else : gogo_pv.loc[index,'t5s'] = "0"

        if index.second % 10 == 0 : gogo_pv.loc[index,'t10s'] = "1"
        else : gogo_pv.loc[index,'t10s'] = "0"

        if index.second % 15 == 0 : gogo_pv.loc[index,'t15s'] = "1"
        else : gogo_pv.loc[index,'t15s'] = "0"

        if index.second % 30 == 0 : gogo_pv.loc[index,'t30s'] = "1"
        else : gogo_pv.loc[index,'t30s'] = "0"

        if index.second == 0 : gogo_pv.loc[index,'t1m'] = "1"
        else : gogo_pv.loc[index,'t1m'] = "0"

        if ( index.second == 0 and index.minute % 5 == 0 ): gogo_pv.loc[index,'t5m'] = "1"
        else : gogo_pv.loc[index,'t5m'] = "0"

        if ( index.second == 0 and index.minute % 10 == 0 ): gogo_pv.loc[index,'t10m'] = "1"
        else : gogo_pv.loc[index,'t10m'] = "0"

        if ( index.second == 0 and index.minute % 30 == 0 ): gogo_pv.loc[index,'t30m'] = "1"
        else : gogo_pv.loc[index,'t30m'] = "0"

        if ( index.second == 0 and index.minute == 0 ): gogo_pv.loc[index,'t1h'] = "1"
        else : gogo_pv.loc[index,'t1h'] = "0"

        if (index.second == 0 and index.minute == 0 and (index.hour == 0 or index.hour == 12)): gogo_pv.loc[index,'t12h'] = "1"
        else: gogo_pv.loc[index,'t12h'] = "0"

        if (index.second == 0 and index.minute == 0 and index.hour == 0):gogo_pv.loc[index,'t1d'] = "1"
        else: gogo_pv.loc[index,'t1d'] = "0"
    ########################################### data insert ######################################
    feature_columns = list(gogo_pv.columns.difference([
    't5s','t10s','t15s','t30s','t1m','t5m','t10m','t30m','t1h','t12h','t1d'
    ]))
    
    db1= 'pis' 
    host1='172.17.48.31' 
    port1= 8086 
    user1='python' 
    passwd1='python123'
    insert_data = DataFrameClient(host1, port1, user1, passwd1, db1)

    grouped = gogo_pv.groupby(['t5s','t10s','t15s','t30s','t1m','t5m','t10m','t30m','t1h','t12h','t1d'])
    for group in grouped.groups:
        tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9, tag10, tag11= group
        tags = dict(t5s=tag1,t10s=tag2,t15s=tag3,t30s=tag4,t1m=tag5,t5m=tag6,t10m=tag7,t30m=tag8,t1h=tag9,t12h=tag10,t1d=tag11)
        sub_df = gogo_pv.groupby(['t5s','t10s','t15s','t30s','t1m','t5m','t10m','t30m','t1h','t12h','t1d']).get_group(group)[feature_columns]
        #insert_data.write_points(sub_df, 'EDGE_TAG_HISTORY_TS', tags=tags)
    ##############################################################################################
    print("Insert Data Time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

schedule.every(2).seconds.do(job)

while 1:
    schedule.run_pending() 
    time.sleep(1)


# In[ ]:




