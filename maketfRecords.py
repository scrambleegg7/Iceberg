#
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io
import tensorflow as tf
from env import setEnv
from time import time

from Iceberg.keras.load_data import load_data

def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
    values: A scalar or list of values.
    Returns:
    a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
    values: A string.
    Returns:
    a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(orig_image_data, image_format, height, width, depth, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded/orig': bytes_feature(orig_image_data),
      #'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/depth': int64_feature(depth),
    }))

def make_train():

    train, test = load_data()

    train_id = train["id"]
    test = test["id"]

    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])

    x1_shape = x_band1.shape
    x2_shape = x_band2.shape


    #scaler = preprocessing.StandardScaler()
    #mm_scaler = preprocessing.MinMaxScaler()
    #stand_x_band1 = mm_scaler.fit_transform(x_band1.reshape(-1,1))
    #stand_x_band2 = mm_scaler.fit_transform(x_band2.reshape(-1,1))
    #x_band1_st = stand_x_band1.reshape(x1_shape)
    #x_band2_st = stand_x_band2.reshape(x2_shape)

    print("check shape .....")
    print(x_band1.shape,x_band2.shape)

    X_train = np.concatenate([x_band1[:, :, :, np.newaxis]
                          , x_band2[:, :, :, np.newaxis]
                         , ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)

    #
    # rescale
    #
    X_train = X_train / 100. +  0.5

    X_angle_train = np.array(train.inc_angle)
    y_train = np.array(train["is_iceberg"])

    return X_train, X_angle_train, y_train


def consolidated2(X_train,y_train):

    tfrecords_filename='tfRecords/train.tfrecords'
    im_width=224
    im_height=224

    if os.path.exists(tfrecords_filename):
        print( tfrecords_filename, "already exists" )
        #return
        print(" rewriting ........")
    start_time = time()
    with tf.Graph().as_default():
        with tf.python_io.TFRecordWriter(tfrecords_filename) as tfrecord_writer:
            print("Saving prepocessed file to '%s'" % tfrecords_filename )
            for img_index in range(X_train.shape[0]):

                if img_index % 100 == 0:
                    print(img_index)

                image_orig_data = X_train[img_index].tostring()
                label_raw = y_train[img_index]
                height, width, depth = X_train.shape[1:]

                example = image_to_tfexample(
                            image_orig_data, "jpg", height, width, depth, label_raw)

                tfrecord_writer.write(example.SerializeToString())

            tfrecord_writer.close()
            print("Preprocessing done in %s seconds" % (time() - start_time))

def main():

    envs = setEnv()

    X_train,X_angle_train, y_train = make_train()
    consolidated2(X_train, y_train)


if __name__ == '__main__':
    main()
    #create_test_data()
