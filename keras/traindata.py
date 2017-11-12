#
from Iceberg.keras.load_data import load_data
from sklearn import preprocessing

import pandas as pd
import numpy as np

def make_train():

    train, test = load_data()

    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])

    x1_shape = x_band1.shape
    x2_shape = x_band2.shape

    #scaler = preprocessing.StandardScaler()
    mm_scaler = preprocessing.MinMaxScaler()
    stand_x_band1 = mm_scaler.fit_transform(x_band1.reshape(-1,1))
    stand_x_band2 = mm_scaler.fit_transform(x_band2.reshape(-1,1))
    x_band1_st = stand_x_band1.reshape(x1_shape)
    x_band2_st = stand_x_band2.reshape(x2_shape)

    print("check shape .....")
    print(x_band1.shape,x_band2.shape,x_band1_st.shape,x_band2_st.shape)


    X_train = np.concatenate([x_band1_st[:, :, :, np.newaxis]
                          , x_band2_st[:, :, :, np.newaxis]
                         , ((x_band1_st+x_band1_st)/2)[:, :, :, np.newaxis]], axis=-1)
    X_angle_train = np.array(train.inc_angle)
    y_train = np.array(train["is_iceberg"])

    return X_train, X_angle_train, y_train

def make_test():

    train, test = load_data()

    ids = test['id'].values

    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])

    x1_shape = x_band1.shape
    x2_shape = x_band2.shape

    #scaler = preprocessing.StandardScaler()
    mm_scaler = preprocessing.MinMaxScaler()
    stand_x_band1 = mm_scaler.fit_transform(x_band1.reshape(-1,1))
    stand_x_band2 = mm_scaler.fit_transform(x_band2.reshape(-1,1))
    x_band1_st = stand_x_band1.reshape(x1_shape)
    x_band2_st = stand_x_band2.reshape(x2_shape)

    print("check shape .....")
    print(x_band1.shape,x_band2.shape,x_band1_st.shape,x_band2_st.shape)

    X_test = np.concatenate([x_band1_st[:, :, :, np.newaxis]
                          , x_band2_st[:, :, :, np.newaxis]
                         , ((x_band1_st+x_band1_st)/2)[:, :, :, np.newaxis]], axis=-1)
    X_angle_test = np.array(test.inc_angle)

    return X_test, X_angle_test, ids
