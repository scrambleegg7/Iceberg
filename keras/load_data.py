
#
import pandas as pd
import numpy as np

from Iceberg import utils
from Iceberg.env import setEnv

import os


def load_data():

    np.random.seed(1017)
    env = setEnv()
    #Load data
    data_dir = env["data_dir"]

    train_data = os.path.join(data_dir,"train.json")
    test_data = os.path.join(data_dir,"test.json")

    print("load data...")
    train = pd.read_json(train_data)
    test = pd.read_json(test_data)

    #train.inc_angle = train.inc_angle.replace('na', 0)
    #train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
    print("angle filled with median.....")
    angle = train.inc_angle.copy()
    angle = angle.replace('na',np.nan)

    angle_median = angle.median()
    angle_mean = angle.mean()
    angle = angle.astype(np.float32).fillna(angle_median)
    train.inc_angle = angle
    print("done!")

    return train, test

def main():
    load_data()


if __name__ == "__main__":
    main()
