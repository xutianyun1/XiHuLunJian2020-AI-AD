# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def km(n_clusters,df,features):
    kmeans_model = KMeans(n_clusters=n_clusters,
                          random_state=2020,
                         ).fit(df[features].values)
    labels = kmeans_model.labels_
    score = metrics.silhouette_score(df[features], labels, metric='euclidean')
    print(n_clusters, 'score is ', score)
    return labels



def addtion_func(addtion_path):
    labelencoding_df_dic = {}
    addition = pd.read_csv(addtion_path)
    addition_copy = addition.copy()
    del addition_copy['requestUrl']
    features = [col for col in addition_copy.columns if col not in ['eventId']]

    str_features = ['requestMethod', 'requestUrlQuery', 'requestBody', 'httpReferer',
                    'accessAgent', 'requestHeader', 'requestContentType', 'responseCode']

    #  labelencoder
    for col in tqdm(str_features):
        labelencoder = LabelEncoder()
        #  缺省值用众数填充
        addition_copy[col] = addition_copy[col].fillna(addition_copy[col].mode())
        #  labelencoder
        addition_copy[col] = labelencoder.fit_transform(addition_copy[col].astype(str))
        labelencoding_df_dic[col] = labelencoder.classes_

    addition_copy['requestBody_len'] = addition_copy['requestBody'].apply(lambda x: len(str(x)))
    #  标准化
    for col in tqdm(addition_copy.columns):
        sta = StandardScaler()
        addition_copy[col] = sta.fit_transform(addition_copy[[col]].values)

    #  聚类
    features = [col for col in addition_copy.columns if col not in ['eventId']]
    for i in tqdm([3]):
        labels = km(i, addition_copy, features)
        submit = addition[['eventId']].copy()
        submit['label'] = labels
        submit.to_csv('../../addition/头给你打歪_addition.csv', index=False)

if __name__ == '__main__':
    addtion_path = '../../data/addition.csv' # 该路径仅供参考
    addtion_func(addtion_path)
