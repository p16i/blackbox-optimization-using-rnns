import numpy as np
import tensorflow as tf


# GP Kernels
def rbf_kernel(np_or_tf, x1,x2,l):
	if np_or_tf == "np":
		return np.exp(-1.0/l**2*np.sum((np.expand_dims(x1,axis=2) - np.expand_dims(x2,axis=1))**2, axis = 3))
	else:
		return tf.exp(-1.0/l**2*tf.reduce_sum((tf.expand_dims(x1,axis=2) - tf.expand_dims(x2,axis=1))**2, axis = 3))

def matern32_kernel(np_or_tf, x1,x2,l,gamma=1.0):
	if np_or_tf == "np":
		dist = np.sum(np.abs(np.expand_dims(x1,axis=2) - np.expand_dims(x2,axis=1)), axis = 3)
		return (1+gamma*np.sqrt(3)*dist/l)*np.exp(-gamma*np.sqrt(3)*dist/l)
	else:
		dist = tf.reduce_sum(np.abs(tf.expand_dims(x1,axis=2) - tf.expand_dims(x2,axis=1)), axis = 3)
		return (1+gamma*np.sqrt(3.0)*dist/l)*tf.exp(-gamma*np.sqrt(3.0)*dist/l)

def matern52_kernel(np_or_tf, x1,x2,l,gamma=1.0):
	if np_or_tf == "np":
		dist = np.sum(np.abs(np.expand_dims(x1,axis=2) - np.expand_dims(x2,axis=1)), axis = 3)
		return (1+gamma*np.sqrt(5)*dist/l+gamma**2*5/3*(dist/l)**2)*np.exp(-gamma*np.sqrt(5)*dist/l)
	else:
		dist = tf.reduce_sum(np.abs(tf.expand_dims(x1,axis=2) - tf.expand_dims(x2,axis=1)), axis = 3)
		return (1+gamma*np.sqrt(5)*dist/l+gamma**2*5/3*(dist/l)**2)*tf.exp(-gamma*np.sqrt(5)*dist/l)

# GP Function
def GP(np_or_tf, X,A,x, l, kernel):
	if np_or_tf == "np":
		k_xX = kernel(np_or_tf, x,X,l)
		return np.squeeze(np.matmul(k_xX,  A),axis=(2,))
	else:
		k_xX = kernel(np_or_tf, tf.expand_dims(x, axis = 1),X,l)
		return tf.squeeze(tf.matmul(k_xX,  A),axis=(2,))


def normalize(minv, maxv, y):
    return 2*(y-minv)/(maxv-minv)-1.0

# Objective Priors
def normalized_gp_function(np_or_tf, X, A, minv, maxv, l, kernel, x):
	return normalize(minv,maxv,GP(np_or_tf, X, A, x, l, kernel))

def un_normalized_gp_function(X,A,minv,maxv,l,kernel,x):
    return GP(np_or_tf, X, A, x, l, kernel)

def airfoil_prior(np_or_tf, X,A,minv,maxv,l,kernel,x):
	if np_or_tf == "np":
		minv = np.tanh(1.5*minv+0.3)
		maxv = np.tanh(1.5*maxv+0.3)
		return  normalize(minv,maxv,np.tanh(1.5*(GP(np_or_tf, X,A,x,l,kernel))+0.3))
	else:
		minv = tf.tanh(1.5*minv+0.3)
		maxv = tf.tanh(1.5*maxv+0.3)
		return  normalize(minv,maxv,tf.tanh(1.5*(GP(np_or_tf, X,A,x,l,kernel))+0.3))

def benchmark_prior0(np_or_tf,X,A,minv,maxv,l,kernel,x):
    if np_or_tf == "np":
        return np.minimum(np.sum((x-X[:,0,:][:,np.newaxis,:])**2, axis = (2))-1,1)
    else:
        return tf.expand_dims(tf.minimum(tf.reduce_sum((x-X[:,0,:])**2, axis = 1)-1,1),1)

def benchmark_prior1(np_or_tf,X,A,minv,maxv,l,kernel,x):

    coeff = 8.0


    if np_or_tf == "np":
        return np.tanh(coeff*(np.maximum(0.0,normalize(minv,maxv,(GP(np_or_tf, X,A,x,l,rbf_kernel)))) \
                +np.minimum(0.0,-normalize(minv,maxv,(GP(np_or_tf, X,A,x,l,matern32_kernel)))) \
                +np.maximum(0.0,-normalize(minv,maxv,(GP(np_or_tf, X,A,x,l,rbf_kernel)))) \
                +np.minimum(0.0,normalize(minv,maxv,(GP(np_or_tf, X,A,x,l,matern32_kernel))))))
    else:
        return  tf.tanh(coeff*(tf.maximum(0.0,normalize(minv,maxv,(GP(np_or_tf, X,A,x,l,rbf_kernel)))) \
                + tf.minimum(0.0,-normalize(minv,maxv,(GP(np_or_tf, X,A,x,l,matern32_kernel)))) \
                + tf.maximum(0.0,-normalize(minv,maxv,(GP(np_or_tf, X,A,x,l,rbf_kernel)))) \
                + tf.minimum(0.0,normalize(minv,maxv,(GP(np_or_tf, X,A,x,l,matern32_kernel))))))

def benchmark_prior2(np_or_tf,X,A,minv,maxv,l,kernel,x):

	if np_or_tf == "np":
		return np.maximum(0.0,normalize(minv,maxv,(GP(np_or_tf, X,A,x,l-0.1,rbf_kernel)))) \
				+np.minimum(0.0,-normalize(minv,maxv,(GP(np_or_tf, X,A,x,l+0.1,matern32_kernel))))

	else:
		return  tf.maximum(0.0,normalize(minv,maxv,(GP(np_or_tf, X,A,x,l-0.1,rbf_kernel)))) \
				+ tf.minimum(0.0,-normalize(minv,maxv,(GP(np_or_tf, X,A,x,l+0.1,matern32_kernel))))

def benchmark_prior3(np_or_tf,X,A,minv,maxv,l,kernel,x):

	if np_or_tf == "np":
		return normalize(minv,maxv,(GP(np_or_tf, X,A,x*np.cos(3*x),l,rbf_kernel)))

	else:
		return normalize(minv,maxv,(GP(np_or_tf, X,A,x*tf.cos(3*x),l,rbf_kernel)))

def benchmark_prior4(np_or_tf,X,A,minv,maxv,l,kernel,x):

	if np_or_tf == "np":
		return np.round(normalize(minv,maxv,(GP(np_or_tf, X,A,np.round(10*x)/10,l,rbf_kernel)))*10)/10

	else:
		return tf.round(normalize(minv,maxv,(GP(np_or_tf, X,A,tf.round(10*x)/10,l,rbf_kernel)))*10)/10

def benchmark_prior5(np_or_tf,X,A,minv,maxv,l,kernel,x):
	tile = lambda x : (x+np.floor(x)+0.5)/1.5


	if np_or_tf == "np":
		return normalize(minv,maxv,(GP(np_or_tf, X,A,((8*x-np.floor(8*x))-0.5)/2+np.floor(8*x)/8,l,kernel)))

	else:
		return normalize(minv,maxv,(GP(np_or_tf, X,A,((8*x-tf.floor(8*x))-0.5)/2+tf.floor(8*x)/8,l,kernel)))

def benchmark_prior6(np_or_tf,X,A,minv,maxv,l,kernel,x):
	tile = lambda x : (x+np.floor(x)+0.5)/1.5


	if np_or_tf == "np":
		return normalize(minv,maxv,(GP(np_or_tf, X,A,0.5*np.abs(x)+((4*x-np.floor(4*x))-0.5)/2+np.floor(4*x)/4-0.5,l,matern32_kernel)))

	else:
		return normalize(minv,maxv,(GP(np_or_tf, X,A,0.5*tf.abs(x)+((4*x-tf.floor(4*x))-0.5)/2+tf.floor(4*x)/4-0.5,l,matern32_kernel)))

def kernel_function(kernel):
    kernel_func = None
    if kernel is "rbf":
        kernel_func = rbf_kernel
    elif kernel is "matern32":
        kernel_func = matern32_kernel
    elif kernel is "matern52":
        kernel_func = matern52_kernel

    return kernel_func

def dataset_function(name):
    func = None
    if name is "normal":
        func = normalized_gp_function
    elif name is "prior0":
        func = benchmark_prior0
    elif name is "prior1":
        func = benchmark_prior1
    elif name is "prior3":
        func = benchmark_prior3

    return func
