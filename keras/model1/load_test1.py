
#
import pandas as pd
import numpy as np

from Iceberg import utils

def test():

    np.random.seed(1017)
    target = 'is_iceberg'

    #Load data
    floc = "/Users/donchan/Documents/myData/KaggleData/Iceberg/data/processed"
    train, train_bands = utils.read_jason(file='train.json', loc=floc)
    test, test_bands = utils.read_jason(file='test.json', loc=floc)

    #target
    train_y = train[target].values
    split_indices = train_y.copy()

    #data set
    train_X = utils.rescale(train_bands)
    train_meta = train['inc_angle'].values
    test_X_dup = utils.rescale(test_bands)
    test_meta = test['inc_angle'].values



def main():
    test()


if __name__ == "__main__":
    main()
