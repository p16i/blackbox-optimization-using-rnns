import imp
import numpy as np

bm = imp.load_source('benchmarkfunctions', '../../benchmarkfunctions.py')

def main(job_id, params):

    x = [ params['x%d'%i][0] for i in range(1,4) ]

    y = bm.hartmann3(x)
    y = float(y)

    x = ','.join( map(lambda x: str(x), x ) )

    print ('job #%2d : (%s)\t->\ty=%f' % ( job_id, x, y))

    return y
