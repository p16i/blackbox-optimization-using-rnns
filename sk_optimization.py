import skopt
import warnings
import numpy as np

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

def get_samples_sk(X,A,minv,maxv, l, dim, n_steps, function, kernel, n, x_start):
	# performs skopt optimization for the first n gp-functions specified by the parameters X,A,minv,maxv
	
	# the i-th gp-function
	fun = lambda x, i : np.asscalar(function("np", X[np.newaxis,i], A[np.newaxis,i],\
							  minv[np.newaxis,i], maxv[np.newaxis,i], l, kernel, \
							  np.array(x)[np.newaxis,np.newaxis,:]))	
	
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		fxn()
		
		samples_sk_x = []
		samples_sk_y = []
		for i in range(n):
			res = skopt.gp_minimize(lambda x: fun(x,i), [(-1.0, 1.0)]*dim, n_calls=n_steps, x0=x_start)
			samples_sk_x += [np.array(res.x_iters)]
			samples_sk_y += [np.array(res.func_vals)]
			
	return np.array(samples_sk_x), np.array(samples_sk_y)
	
def save_samples(samples_x, samples_y, directory):
	np.save( directory + "/samples_x", samples_x)
    np.save( directory + "/samples_y", samples_y)
