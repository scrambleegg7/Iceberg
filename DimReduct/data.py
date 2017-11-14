from multiprocessing import Pool
from tqdm import tqdm
import gc
#
import numpy as np # linear algebra
import pandas as pd # data processing
import datetime as dt
#
from random import choice, sample, shuffle, uniform, seed
from math import exp, expm1, log1p, log10, log2, sqrt, ceil, floor, isfinite, isnan
from itertools import combinations
#import for image processing
import cv2
from scipy.stats import kurtosis, skew
from scipy.ndimage import laplace, sobel
#evaluation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
import xgboost as xgb
import lightgbm as lgb

from Iceberg.env import setEnv
import os

###############################################################################
def read_jason(file='', loc='../input/'):

    filename = os.path.join(loc,file)
    df = pd.read_json(filename)
    df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)
    #print(df['inc_angle'].value_counts())

    band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)

    bands = np.stack((band1, band2,  0.5 * (band1 + band2)), axis=-1)
    del band1, band2

    return df, bands

###############################################################################


def run_lgb(params={}, lgb_train=None, lgb_valid=None, lgb_test=None, test_ids=None, nr_round=2000, min_round=100, file=''):

    print('\nLightGBM: {}'.format(params['boosting']))
    model2 = lgb.train(params,
                       lgb_train,
                       nr_round,
                       lgb_valid,
                       verbose_eval=50, early_stopping_rounds=min_round)

    pred = model2.predict(lgb_test, num_iteration=model2.best_iteration)
    #
    subm = pd.DataFrame({'id': test_ids, 'is_iceberg': pred})
    subm.to_csv(file, index=False, float_format='%.6f')
    #
    df = pd.DataFrame({'feature':model2.feature_name(), 'importances': model2.feature_importance()})

    return pred, df

###############################################################################
#forked from
#https://www.kaggle.com/the1owl/planet-understanding-the-amazon-from-space/natural-growth-patterns-fractals-of-nature/notebook
def img_to_stats(paths):

    img_id, img = paths[0], paths[1]

    #ignored error
    np.seterr(divide='ignore', invalid='ignore')

    bins = 20
    scl_min, scl_max = -50, 50
    opt_poly = True
    #opt_poly = False

    try:
        st = []
        st_interv = []
        hist_interv = []
        for i in range(img.shape[2]):
            img_sub = np.squeeze(img[:, :, i])

            #median, max and min
            sub_st = []
            sub_st += [np.mean(img_sub), np.std(img_sub), np.max(img_sub), np.median(img_sub), np.min(img_sub)]
            sub_st += [(sub_st[2] - sub_st[3]), (sub_st[2] - sub_st[4]), (sub_st[3] - sub_st[4])]
            sub_st += [(sub_st[-3] / sub_st[1]), (sub_st[-2] / sub_st[1]), (sub_st[-1] / sub_st[1])] #normalized by stdev
            st += sub_st
            #Laplacian, Sobel, kurtosis and skewness
            st_trans = []
            st_trans += [laplace(img_sub, mode='reflect', cval=0.0).ravel().var()] #blurr
            sobel0 = sobel(img_sub, axis=0, mode='reflect', cval=0.0).ravel().var()
            sobel1 = sobel(img_sub, axis=1, mode='reflect', cval=0.0).ravel().var()
            st_trans += [sobel0, sobel1]
            st_trans += [kurtosis(img_sub.ravel()), skew(img_sub.ravel())]

            if opt_poly:
                st_interv.append(sub_st)
                #
                st += [x * y for x, y in combinations(st_trans, 2)]
                st += [x + y for x, y in combinations(st_trans, 2)]
                st += [x - y for x, y in combinations(st_trans, 2)]

            #hist
            #hist = list(cv2.calcHist([img], [i], None, [bins], [0., 1.]).flatten())
            hist = list(np.histogram(img_sub, bins=bins, range=(scl_min, scl_max))[0])
            hist_interv.append(hist)
            st += hist
            st += [hist.index(max(hist))] #only the smallest index w/ max value would be incl
            st += [np.std(hist), np.max(hist), np.median(hist), (np.max(hist) - np.median(hist))]

        if opt_poly:
            for x, y in combinations(st_interv, 2):
                st += [float(x[j]) * float(y[j]) for j in range(len(st_interv[0]))]

            for x, y in combinations(hist_interv, 2):
                hist_diff = [x[j] * y[j] for j in range(len(hist_interv[0]))]
                st += [hist_diff.index(max(hist_diff))] #only the smallest index w/ max value would be incl
                st += [np.std(hist_diff), np.max(hist_diff), np.median(hist_diff), (np.max(hist_diff) - np.median(hist_diff))]

        #correction
        nan = -999
        for i in range(len(st)):
            if isnan(st[i]) == True:
                st[i] = nan

    except:
        print('except: ')

    return [img_id, st]


def extract_img_stats(paths):
    imf_d = {}
    p = Pool(8) #(cpu_count())
    ret = p.map(img_to_stats, paths)
    for i in tqdm(range(len(ret)), miniters=100):
        imf_d[ret[i][0]] = ret[i][1]

    ret = []
    fdata = [imf_d[i] for i, j in paths]
    return np.array(fdata, dtype=np.float32)


def process(df, bands):

    data = extract_img_stats([(k, v) for k, v in zip(df['id'].tolist(), bands)]); gc.collect()
    data = np.concatenate([data, df['inc_angle'].values[:, np.newaxis]], axis=-1); gc.collect()

    print(data.shape)
    return data




###############################################################################
if __name__ == '__main__':

    np.random.seed(1017)
    target = 'is_iceberg'

    env = setEnv()
    fileloc = env['data_dir']
    print(fileloc)
    #Load data
    train, train_bands = read_jason(file='train.json', loc=fileloc)
    test, test_bands = read_jason(file='test.json', loc=fileloc)

    train_X = process(df=train[:10], bands=train_bands)
    #train_y = train[target].values

    print(train_X[0])
