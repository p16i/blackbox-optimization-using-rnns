import yaml
import math
import numpy as np
import time

functions = {
    'parabola' : lambda x : math.pow(x,2)
}


def loadConfig():
    with open("config.yaml", 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def getDataPath(dim,dataset):
    return "%s/%dD/%s" % (loadConfig()['DATA_DIR'], dim, dataset)

def loadData(dim, dataset):
    path = getDataPath(dim, dataset)
    return np.load(path+"/X.npy"), np.load(path+"/A.npy"), np.load(path+"/minv.npy"), np.load(path+"/maxv.npy")
