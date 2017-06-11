import imp

utils = imp.load_source('utils', '../../utils.py')

def main(job_id, params):
    y = utils.goldstein_price(params['x1'], params['x2'])
    print ('job #%2d : (%2.5f,%2.5f)\t->\ty=%f' % ( job_id, params['x1'], params['x2'], y))

    return float(y)
