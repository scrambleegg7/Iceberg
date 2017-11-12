#
import numpy as np
import pandas as pd

from Iceberg.keras.traindata import make_test

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

def process():

    np.random.seed(1017)
    target = 'is_iceberg'

    X_test, X_angle_test, ids = make_test()
    print(" shape check test data ........")
    print(X_test.shape, X_angle_test.shape)

    file_path = "weights/keras_model2_weights.hdf5"
    print("model file to be used for valuation ....", file_path)

    model = get_model()
    print(model.summary())

    if os.path.isfile(file_path):

        model.load_weights(file_path)

        print('\nPredict...') #, flush=True)


        #prediction
        batch_size=32
        pred = model.predict([X_test], batch_size=batch_size, verbose=0)
        #pred = np.squeeze(pred, axis=-1)
        print(pred.shape)
        print(pred[:10])

        d = datetime.now().strftime("%Y%m%d_%H%M%S")

        file = 'subm_{}.csv'.format(d)
        print('\nSave to {}'.format(file))
        subm = pd.DataFrame({'id': ids, target: pred[:,1]})
        subm.to_csv('output/{}'.format(file), index=False, float_format='%.6f')


def main():
    process()



if __name__ == "__main__":
    main()
