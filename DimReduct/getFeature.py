

from multiprocessing import Pool, cpu_count
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import fbeta_score
from PIL import Image, ImageStat
import pandas as pd
import numpy as np
import glob, cv2

from Iceberg.keras.traindata import make_train

def get_features(path):
    try:
        st = []
        #pillow
        #im_stats_ = ImageStat.Stat(Image.open(path))
        im_stats_ = ImageStat.Stat(path)

        #print(im_stats_.sum)
        st += im_stats_.sum
        st += im_stats_.mean
        st += im_stats_.rms
        st += im_stats_.var
        st += im_stats_.stddev
        #cv2
        img = cv2.imread(path)
        bw = cv2.imread(path,0)
        st += list(cv2.calcHist([bw],[0],None,[256],[0,256]).flatten()) #histogram
        m, s = cv2.meanStdDev(img) #mean and standard deviation
        st += list(m)
        st += list(s)
        st += cv2.Laplacian(bw, cv2.CV_64F).var() #blurr
        st += (bw<10).sum()
        st += (bw>245).sum()
        #img = cv2.resize(img, (20, 20), cv2.INTER_LINEAR)
    except:
        print(path)
    return [path, st]

def normalize_img(paths):

    imf_d = {}
    p = Pool(2) #(cpu_count())
    ret = p.map(get_features, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    #fdata = np.array(fdata, dtype=np.uint8)

    return fdata

def main():

    X_train, X_angle_train, y_train = make_train()
    #fdata = normalize_img(X_train[0])
    #st = ImageStat.Stat(X_train[0])

if __name__ == "__main__":
    main()
