import yaml
import math
import time
import pandas as pd
import numpy as np
import re
import glob

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

def string_to_number(st):
    stt = re.sub('^-','',st)
    if re.match('^\d+$', stt):
        return int(st)
    elif re.match('[\d\.]+',stt):
        return float(st)
    return st

def extract_lstm_trained_output(log_dir, debug = False):

    config = loadConfig()
    path = config['BASE_LOG_DIR'] + '/' + log_dir + '/*'

    files = glob.glob(path)
    no_files = len(files)
    if debug :
        print('Found %d files in %s' % (no_files, path))

    results = []
    for fp in files:
        with open(fp) as f:
            columns = dict()
            lines = f.readlines()
            hyper_block = False
            for line in lines:
                reg_ex = '^-+'
                if re.match(reg_ex, line):
                    hyper_block = not hyper_block

                if hyper_block and not re.match(reg_ex, line):
                    k,v = re.split(" *: +", line.strip())
                    columns[k] = string_to_number(v)

            last_lines = lines[-2].replace("Last output: ","")

            for t in last_lines.split('|'):
                k,v = re.split(" *: +", t.strip())
                columns[k] = string_to_number(v)
            results.append(columns)
    return pd.DataFrame(results)

'''
Return RNN Scope (aka model name) of trained model for particular
dimension, kernel, loss and no. of hidden units
'''
def get_trained_model(dim, kernel, loss, no_hidden_unit=100):
    df = extract_lstm_trained_output('%dd-%s' % (dim, kernel) )
    return df.loc[ (df['Number of hidden Units'] == no_hidden_unit)  & (df['Loss'] == loss) ]['RNN Scope'].values[0]
