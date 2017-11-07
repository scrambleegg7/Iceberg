#
# zillow Data path
#

import numpy as np
import os

from sys import platform

from os.path import join

def setEnv():

    envs = {}
    if platform == "linux":
        envs["data_dir"] = "/home/donchan/Documents/myData/KaggleData/Iceberg/data/processed"
    else:
        envs["data_dir"] = "/Users/donchan/Documents/myData/KaggleData/Iceberg/data/processed"

    return envs
