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

def goldstein_price(x_1,x_2):
    #  https://www.sfu.ca/~ssurjano/goldpr.html
    # f(x*) = 3, x* = (0,-1) and x_i = [-2,2]

    y = (1 + (x_1 + x_2 + 1)**2 * (19 - 14*x_1 + 3*x_1**2 - 14*x_2 + 6*x_1*x_2+ 3*x_2**2))*\
    (30 + (2*x_1-3*x_2)**2 * (18 - 32*x_1 + 12*x_1**2 + 48*x_2 - 36*x_1*x_2+ 27*x_2**2))

    return y


def parse_spearmint_run_log(path, xcols, run_id="something" ):
    with open(path) as f:
        data = []
        for l in f.readlines():
            output = float(re.findall(r"y=(.+)", l)[0])
            job_id = int(re.findall(r"job #\s*(\d+)", l)[0])
            xs = map( lambda x : float(x), re.findall(r"\((.+)\)", l)[0].split(',') )
            data.append((run_id,job_id,output,*xs))
        return pd.DataFrame(data, columns= ['job_id', 'job_run', 'y'] + xcols)

def retrieve_spearmint_runs_from_dir(dirname, xcols, runs_dir="runs"):
    dir_path = "%s/%s/%s/"  % ( loadConfig()['SPEARMINT_CONF_BASE_DIR'], dirname, runs_dir )
    data = []
    for f in listdir(dir_path):
        if not isfile(join(dir_path, f)):
            continue

        df = parse_spearmint_run_log( '%s/%s' % (dir_path, f ), xcols, f)

        data.append(df)
    return data

def combine_spearmint_results(res, step=21):
    data  = []
    for r in res:
        if len(r) >= step:
            data.append(r.y[:step])
    return np.array(data)
