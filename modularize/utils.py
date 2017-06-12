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

def goldstein_price(x_1,x_2):
    #  https://www.sfu.ca/~ssurjano/goldpr.html
    # f(x*) = 3, x* = (0,-1) and x_i = [-2,2]

    y = (1 + (x_1 + x_2 + 1)**2 * (19 - 14*x_1 + 3*x_1**2 - 14*x_2 + 6*x_1*x_2+ 3*x_2**2))*\
    (30 + (2*x_1-3*x_2)**2 * (18 - 32*x_1 + 12*x_1**2 + 48*x_2 - 36*x_1*x_2+ 27*x_2**2))

    return y
