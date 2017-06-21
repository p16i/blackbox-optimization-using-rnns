import imp
import numpy as np

bm = imp.load_source('benchmarkfunctions', '../../benchmarkfunctions.py')

def main(job_id, params):
    y = bm.goldstein_price([params['x1'][0], params['x2'][0]])
    y = float(y)
    print ('job #%2d : (%2.5f,%2.5f)\t->\ty=%f' % ( job_id, params['x1'], params['x2'], y))

    return y
