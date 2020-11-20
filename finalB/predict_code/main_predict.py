# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
import joblib
#  特征工程
def feature_gen(test_df):
    #  浅拷贝一份数据，不在原有数据上做改动
    test = test_df.copy()

    #  删除不用的特征
    #  包括数据清洗中的缺省值过多、字段唯一、字段分散以及特征筛选后
    useless = ['srcGeoCity', 'srcGeoAddress', 'srcGeoLatitude', 'destGeoAddress',
               'srcGeoLatitude', 'srcGeoLongitude', 'appProtocol', 'transProtocol',
                'destGeoAddress', 'requestHeader', 'responseHeader', 'name', 'srcPort', ]
    test.drop(useless, axis=1, inplace=True)

    #  区分类别特征、数字特征
    numeric = ['bytesIn', 'bytesOut', 'destGeoLatitude', 'destGeoLongitude', 'destPort',
               'responseCode', 'txId', 'requestHeader_len', 'responseHeader_len', ]

    categoric = [n for n in list(test.columns) if n not in numeric]
    categoric.remove('eventId')

    #  填充缺省值
    test[categoric] = test[categoric].fillna('Nan')

    def bfs(N, target):
        '''
        二分查找
        :param N:
        :param target:
        :return:
        '''
        left, right = 0, len(N) - 1
        while left < right:
            mid = (left + right) >> 1
            if N[mid] > target:
                right = mid - 1
            elif N[mid] < target:
                left = mid + 1
        return N[left]

    def encoder(x, feature_encoder,  encoder_dict):
        #  并未在训练集中出现过的新字段，进行处理
        if x in encoder_dict.keys():
            return encoder_dict[x]
        return encoder_dict[bfs(feature_encoder[n].values.tolist(), x)]

    print('+++++++predict ... +++++++')
    #  对类别特征进行labelencoder编码
    for n in categoric:
        feature_encoder = pd.read_csv('feature_encoder/'+n+'.csv')
        encoder_dict = dict()
        for index, row in feature_encoder.iterrows():
            encoder_dict[row[n]] = row['encoder']
        test[n] = test[n].apply(lambda x: encoder(x, feature_encoder, encoder_dict))

    x_test = test
    return x_test


def test_func(test_path, save_path):
    submission = pd.DataFrame()
    #  读取测试集
    test = pd.read_csv(test_path)
    #  特征工程
    x_test = feature_gen(test)
    test_pred = np.zeros((x_test.shape[0], ))
    #  lgbm五折
    for i in range(1, 6):
        model_name = 'lgbm_model_'+str(i)+'.pkl'
        lgbm_model = joblib.load('model_pkl/'+model_name)
        pred_proba = lgbm_model.predict_proba(x_test, num_iteration=lgbm_model.best_iteration_)
        test_pred += [n[1]/5 for n in pred_proba]

    add_test_pred = test_pred

    submission['eventId'] = test['eventId']
    submission['label'] = [1 if i > 0.5 else 0 for i in add_test_pred]
    print(submission['label'].value_counts())
    #  保存预测结果
    submission[['eventId', 'label']].to_csv(save_path + '头给你打歪_finalA.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    time_start = time.time()
    test_path = '../../data/test_1.csv'
    sava_path = '../../finalA/'
    test_func(test_path, sava_path)
    time_end = time.time()
    print('time cost: ', time_end-time_start, 's')