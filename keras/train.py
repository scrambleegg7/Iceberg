#
import numpy as np
import pandas as pd

from Iceberg.keras.traindata import make_train

from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from Iceberg.keras.model.kmodel01 import get_model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import os
from datetime import datetime

from sklearn.metrics import log_loss

def get_callbacks(filepath, patience=100):
    es = EarlyStopping('val_loss', patience=patience, mode="min", verbose=0, min_delta=1e-4,)
    #msave = ModelCheckpoint(filepath, save_best_only=True)
    msave = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, save_weights_only=True, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=40, verbose=0, epsilon=1e-4, mode='min')

    return [es, msave, reduce_lr_loss]

file_path = "weights/keras_model2_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)


def process():

    X_train, X_angle_train, y_train = make_train()
    print(" shape check ........")
    print(X_train.shape, X_angle_train.shape, y_train.shape)

    y_onehot = np_utils.to_categorical( y_train )
    num_classes = y_onehot.shape[1]

    n_folds = 5
    batch_size = 32
    epochs = 50 ## change this to 80
    #steps_per_epoch=np.power(2,14) /batch_size ## change to 2^14

    model = get_model()
    print(model.summary())

    print("keep initial model weights for CV ....")
    weights_init = "weights/weights_init.hdf5"
    model.save(weights_init)

    #kf = StratifiedShuffleSplit(n_splits=nr_runs, test_size=test_ratio, train_size=None, random_state=split_seed)
    skf = StratifiedKFold( n_splits=n_folds, shuffle=True, random_state=42)

    #training, evaluation, test and make submission
    for r, (train_index, valid_index) in enumerate(skf.split(X_train,y_train)):

        tmp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        print("Fold: {} time: {}".format(r,tmp))
        x_train, x_valid = X_train[train_index], X_train[valid_index]
        #y_train_oh, y_valid_oh = y_onehot[train_index,:], y_onehot[valid_index,:]
        y_train_oh, y_valid_oh = y_train[train_index], y_train[valid_index]

        print('splitted: {0}, {1}'.format(x_train.shape, x_valid.shape)) #, flush=True)
        print('splitted: {0}, {1}'.format(y_train_oh.shape, y_valid_oh.shape)) #, flush=True)
        ################################
        if r > 0:
            model.load_weights(weights_init)

        #optimizer = Adam(lr=0.001)
        #optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002)

        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        #model.compile(loss="categorical_crossentropy", opimizer=optimizer, metrics=["accuracy"])

        gen_images = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.3,
                height_shift_range=0.3,
                shear_range=0.2,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,)

        model.fit_generator(
                gen_images.flow(x_train,y_train_oh,batch_size=batch_size),
                steps_per_epoch=np.ceil(32.0 * float(y_train_oh.shape[0]) / float(batch_size)),
                epochs=epochs,
                validation_data=gen_images.flow(x_valid,y_valid_oh,batch_size=2*batch_size),
                validation_steps=np.ceil(32.0 * float(y_valid_oh.shape[0]) / float(batch_size)),
                callbacks=callbacks)

        if os.path.isfile(file_path):

            model.load_weights(file_path)
            p = model.predict([x_train], batch_size=batch_size, verbose=1)
            print('\n\nEvaluate loss in training data: {}'.format(log_loss(y_train_oh, p)))

            p = model.predict([x_valid], batch_size=batch_size, verbose=1)
            print('\n\nEvaluate loss in validation data: {}'.format(log_loss(y_valid_oh, p)))

            print('\nPredict...') #, flush=True)





def main():
    process()



if __name__ == "__main__":
    main()
