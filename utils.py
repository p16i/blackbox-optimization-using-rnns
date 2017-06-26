import yaml
import math
import time
import pandas as pd
import numpy as np
import re

from os import listdir
from os.path import isfile, join


functions = {
    'parabola' : lambda x : math.pow(x,2)
}


def loadConfig():
    with open("config.yaml", 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def getDataPath(dim,dataset,kernel):
    return "%s/%dD/%s/%s" % (loadConfig()['DATA_DIR'], dim, kernel, dataset)

def loadData(dim, dataset, kernel ):
    path = getDataPath(dim, dataset, kernel=kernel)
    return np.load(path+"/X.npy"), np.load(path+"/A.npy"), np.load(path+"/minv.npy"), np.load(path+"/maxv.npy")

def min_up_to_k(data):
    total = len(data)
    mins_up_to_k = [data[0]]

    for k in range(1, total):
        m = mins_up_to_k[k-1]
        if mins_up_to_k[k-1] > data[k]:
            m = data[k]
        mins_up_to_k.append(m)

    return mins_up_to_k
