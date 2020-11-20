# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import joblib
import lightgbm as lgb
from sklearn.metrics import f1_score
import xgboost as xgb

def feature_gen(train_df, model='lgbm'):
    '''
    数据清洗以及特征工程
    :param train_df:
    :param model:
    :return:
    '''
    #  浅拷贝一份数据，不在原有数据上做改动
    train = train_df.copy()

    #  删除不用的特征
    #  包括数据清洗中的缺省值过多、字段唯一、字段分散以及特征筛选后
    useless = ['srcGeoCity', 'srcGeoAddress', 'srcGeoLatitude', 'destGeoAddress',
               'srcGeoLatitude', 'srcGeoLongitude', 'appProtocol', 'transProtocol',
               'destGeoAddress', 'requestHeader', 'responseHeader', 'name', 'srcPort',
               ]
    train.drop(useless, axis=1, inplace=True)

    #  区分类别特征、数字特征
    numeric = ['bytesIn', 'bytesOut', 'destGeoLatitude', 'destGeoLongitude', 'destPort',
               'responseCode', 'txId', ]
    categoric = [n for n in list(train.columns) if n not in numeric]
    categoric.remove('label')
    categoric.remove('eventId')

    #  填充缺省值
    train[categoric] = train[categoric].fillna('Nan')
    #  对类别特征进行labelencoder编码
    for n in categoric:
        labelencoder = LabelEncoder()
        labelencoder_df = pd.DataFrame()
        labelencoder.fit(train[n])
        mode = train[n].mode().values[0]
        train[n] = labelencoder.transform(train[n])
        #  保存一份文件供测试集特征工程使用
        labelencoder_df[n] = labelencoder.classes_
        labelencoder_df['encoder'] = labelencoder.transform(labelencoder_df[n])
        labelencoder_df['mode_is'] = 0
        model_index = labelencoder_df[labelencoder_df[n] == mode].index.values[0]
        labelencoder_df.loc[model_index, 'mode_is'] = 1
        labelencoder_df.to_csv('../../finalB/predict_code/feature_encoder/'+n+'.csv', index=False)

    y_train = train['label']
    del train['label']
    x_train = train
    return x_train, y_train


def lgbm_model(x_train, y_train):
    '''
    lgbm模型训练
    :param x_train:
    :param y_train:
    :return:
    '''
    #  5折交叉验证
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
    #  超参数
    model = lgb.LGBMClassifier(
        objective='binary',  # 定义的目标函数
        # objective= f1_loss,
        num_leaves=70,
        n_estimators=250,

        max_depth=7,
    )
    i = 0
    train_scores, val_scores = list(), list()
    for trn_idx, val_idx in folds.split(x_train, y_train):
        #  划分训练集和验证集
        train_x = x_train.iloc[trn_idx]
        train_y = y_train.iloc[trn_idx]
        val_x = x_train.iloc[val_idx]
        val_y = y_train.iloc[val_idx]
        #  训练模型
        model.fit(train_x,
                  train_y,
                  eval_set=[(train_x, train_y), (val_x, val_y)],
                  eval_metric={'auc'},
                  early_stopping_rounds=20, )
        #  对训练集预测
        train_f1_score = f1_score(train_y, model.predict(train_x, num_iteration=model.best_iteration_), average='micro')
        #  对验证集预测
        val_f1_score = f1_score(val_y, model.predict(val_x, num_iteration=model.best_iteration_), average='micro')
        train_scores.append(train_f1_score)
        val_scores.append(val_f1_score)
        print('train_f1_score: ', train_f1_score)
        print('val_f1_score: ', val_f1_score)
        #  保存模型
        i += 1
        joblib.dump(model, '../../finalB/predict_code/model_pkl/lgbm_model_' + str(i) + '.pkl')
    print('---------------------------------------------')
    print('train_f1_mean_score: ', np.mean(train_scores))
    print('val_f1_mean_score: ', np.mean(val_scores))


def train_func(train_path):
    #  载入训练集和测试集
    train = pd.read_csv(train_path)
    #  特征工程
    x_train, y_train = feature_gen(train)
    print(x_train.shape)
    #  lgbm
    print('+++++++lgbm模型训练++++++++')
    lgbm_model(x_train, y_train)


if __name__ == '__main__':
    train_path = '../../data/train.csv'
    train_func(train_path)
