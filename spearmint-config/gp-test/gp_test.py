import numpy as np
import imp
import os


utils = imp.load_source('utils', '../../utils.py')

DIM = 2
N_BUMPS = 6
LENGTH = 2.0/N_BUMPS*np.sqrt(DIM)

data_idx = int(os.environ['GP_TEST_INDEX'])
print('Loading GP-Test %dth' % data_idx)

x_2d, a_2d, min_2d, max_2d = utils.loadData(2, 'testing')
X = np.expand_dims(x_2d[data_idx], axis=0)
A = a_2d[data_idx]
miv = min_2d[data_idx]
mxv = max_2d[data_idx]

def kernel(x1,x2):
    return np.exp(-1.0/LENGTH**2*np.sum((np.expand_dims(x1,axis=2) - np.expand_dims(x2,axis=1))**2, axis = 3))


def GP(X,A,x):
    k_xX = kernel(x,X)
    return np.squeeze(np.matmul(k_xX,  A),axis=(2,))

def normalize(minv, maxv, y):
    return 2*(y-minv)/(maxv-minv)-1.0


def evaluate_gp_value(x1,x2):
    xx = np.array([x1,x2]).reshape(1,1,2);
    z = GP(X,A, xx).T
    z_norm = normalize(miv, mxv, z)
    return float(z_norm)


def main(job_id, params):
    y = evaluate_gp_value(params['x1'], params['x2'])
    print ('job #%2d : (%2.5f,%2.5f)\t->\ty=%f' % ( job_id, params['x1'], params['x2'], y))

    return y
