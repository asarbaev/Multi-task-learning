
# coding: utf-8

# In[32]:


#There were providen 4 different .csv files. 
#In this chapter we are going to subsample interactions.csv file by user activity and offer popularity. 
#Thus, we set threshold which is equal 30.
#i.e. we are going to remove those users who have a number of interactions with unique offers less than 30 times
#and the same for offers popularity: we are going to remove offers which were interacted by unique users less than 30 times.
#After, we are going to take impressions.csv file where were provided information about displayed offers for users and remove all users and offer who are absent in interaction dataset after subsampling.
#Input: interactions.csv and impressions.csv
#Output: single dataset with following structure  -> user_id	item_id	interaction_type	displayed	created_at. 
#Where interaction_type can take values from 0 to 3 (0-non-interacted, 1-clicked, 2-bookmarked,3 replied)
#displyed can take values either 0 or 1 (0 - curent offer didn't show to current user, 1 - otherwise)
#created_at is the number of week

import sys
import csv
import time
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
import datetime

#p - path to the directory of the provided dataset
#th - threshold for subsampling
#s - path to the directory where you want to save the file

def SS_dataset(p,th,s):
    df = pd.read_csv( str ( p ) + '/interactions.csv', header = 0 , sep = '\t' )
    df = df.drop( df [ df.interaction_type == 4 ].index )                                             #remove "delete" type of interactions
    df['created_at'] = pd.to_datetime( df['created_at'], unit = 's' )
    data = df.groupby( ['user_id','item_id'] )['interaction_type'].count().reset_index()           #calculate number of unique pairs(user,item)
    data.interaction_type = np.where( data.interaction_type > 1 , 1 , data.interaction_type ) 
    data = data.groupby( ['item_id'],sort=True ).sum()
    reject_list = data[(data.interaction_type < th)]
    reject_items = list(reject_list.index)
    df = df[~df['item_id'].isin(reject_items)]
    data = df.groupby(['user_id','item_id'])['interaction_type'].count().reset_index()
    data.interaction_type = np.where( data.interaction_type > 1 , 1 , data.interaction_type )
    data = data.groupby(['user_id'],sort=True).sum()
    reject_list = data[(data.interaction_type < th)]
    reject_users = list(reject_list.index)
    df = df[~df['user_id'].isin(reject_users)]
    
    f = open(str(p)+'/impr.csv', 'rb')
    reader = csv.reader(f,delimiter='\t')
    headers = reader.next()
    column = {}
    for h in headers:
        column[h] = []

    for row in reader:
        for h, v in zip(headers, row):
            column[h].append(v)
    for i in range (len(column['items'])):
        column['items'][i]=map(int, column['items'][i].split(","))
    column['user_id']=map(int, column['user_id'])
    list_1=[]
    for i in range (len(column['user_id'])):
        us_list=[column['user_id'][i]]*len(set(column['items'][i]))
        it_list=set(column['items'][i])
        time_list=[column['week'][i]]*len(set(column['items'][i]))
        list_1.append(zip(us_list,it_list,time_list))
    list_1=list(set(list(itertools.chain.from_iterable(list_1))))                                               #took unique pairs
    list_1=[list(tup)+[0]+[1] for tup in list_1]                                                                #convert to list of lists
    df_list_1 = pd.DataFrame(list_1, columns=['user_id','item_id','created_at','interaction_type','displayed']) #df_list_1 is a list of all displyaed pairs from "impressions" file
    df_list_1=df_list_1[['user_id','item_id','interaction_type','displayed','created_at']]
    df.created_at=pd.to_datetime(df.created_at).dt.week
    df_list_1g=df_list_1.groupby(['user_id','item_id']).count().reset_index()
    
    #merge interacted pairs with displayed to define interacted pairs that were not displayed
    df = pd.merge( df, df_list_1g, left_on=['user_id','item_id'],right_on=['user_id','item_id'], how='left', indicator='flag')    
    #df_list_undisp is list of all interacted pairs that weren't displayed
    df_list_undisp=df[df['flag']=='left_only']
    df_list_undisp['displayed'].fillna(0, inplace=True)
    df_list_undisp=df_list_undisp[['user_id','item_id','interaction_type_x','displayed','created_at_x']]
    df_list_undisp.columns=['user_id','item_id','interaction_type','displayed','created_at']
    df_list_undisp['displayed']=df_list_undisp['displayed'].astype(np.int64)
    
    #df is a list of all interacted pairs that were displayed
    df = df[df['flag']=="both"]
    df = df[['user_id','item_id','interaction_type_x','displayed','created_at_x']]
    df.columns = ['user_id','item_id','interaction_type','displayed','created_at']
    df['displayed'] = df['displayed'].astype(np.int64)
    
    #df_list_2g is df_list_2 list that was grouped by user_id,item_id to define which pairs we should drop from impression file
    df_list_2g=df.groupby(['user_id','item_id'],as_index=False).count()
    
    #merge df_list_1, df_list2g to define pairs should be droped from impression file
    df_list_1=pd.merge(df_list_1, df_list_2g, on=['user_id','item_id'], how='left', indicator='flag')
    #all displayed pairs from "impressions" file except of interacted pairs from "interactions file"
    df_list_1=df_list_1[df_list_1['interaction_type_y'].isnull()]
    df_list_1=df_list_1[['user_id','item_id','interaction_type_x','displayed_x','created_at_x']]
    df_list_1.columns=['user_id','item_id','interaction_type','displayed','created_at']
    
    target_list=df_list_1.append([df,df_list_undisp])
    target_list['displayed']=target_list['displayed'].astype(np.int64)
    #combination of 3 lists. Where:
    #1) df - list of all interacted pairs except of non-displayed
    #2) df_list_undisp - list of all non-displayed interacted pairs
    #3) df_list_1 - list of all displayed pairs except of pairs which are included in first two lists.
    target_list.to_csv( str(s)+"/1_target_list.csv",index=None)
    return target_list


# In[33]:


SS_dataset('/Users/amirasarbaev/Downloads/Internship/data',30,"/Users/amirasarbaev/Desktop")

