import numpy as np
import tensorflow as tf

def branin(x):
    a = 1
    b = 5.1 / (4*np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)

    x = 7.5*x+np.array([2.5, 7.5])

    minv = 0.397887
    maxv = 307

    y = a*(x[:,:,1] - b*x[:,:,0]**2 + c*x[:,:,0] - r)**2 + s*(1-t)*np.cos(x[:,:,0]) + s
    y = 2*(y-minv)/(maxv-minv)-1

    return y

def branin_tf(x):
    a = 1
    b = 5.1 / (4*np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)

    x = 7.5*x+tf.constant([2.5, 7.5])

    minv = 0.397887
    maxv = 307

    y = a*(x[:,1] - b*x[:,0]**2 + c*x[:,0] - r)**2 + s*(1-t)*tf.cos(x[:,0]) + s
    y = 2*(y-minv)/(maxv-minv)-1

    return tf.reshape(y, (-1,1))

def goldstein_price(x, normalize= True):

    minv = 3
    maxv = 1020000
    x_1, x_2 = np.split(x,[1],axis=-1)

    y = (1 + (x_1 + x_2 + 1)**2 * (19 - 14*x_1 + 3*x_1**2 - 14*x_2 + 6*x_1*x_2+ 3*x_2**2))*\
    (30 + (2*x_1-3*x_2)**2 * (18 - 32*x_1 + 12*x_1**2 + 48*x_2 - 36*x_1*x_2+ 27*x_2**2))

    if normalize:
        y = 2*(y-minv)/(maxv-minv)-1

    return y

def goldstein_price_tf(x):

    minv = 3
    maxv = 1020000

    y = (1 + (x[:,0] + x[:,1] + 1)**2 * (19 - 14*x[:,0] + 3*x[:,0]**2 - 14*x[:,1] + 6*x[:,0]*x[:,1]\
                                             + 3*x[:,1]**2))*\
    (30 + (2*x[:,0]-3*x[:,1])**2 * (18 - 32*x[:,0] + 12*x[:,0]**2 + 48*x[:,1] - 36*x[:,0]*x[:,1]\
                                        + 27*x[:,1]**2))

    y = 2*(y-minv)/(maxv-minv)-1

    return tf.reshape(y, (-1,1))

def hartmann3(x, normalize=True):

    minv = -3.86278
    maxv = 18.06

    x = np.array(x)

    if normalize:
        x = (x+1)/2

    A = np.array([[3.0, 10, 30],[0.1, 10, 35],[3.0, 10, 30],[0.1, 10, 35]])
    P = 0.0001 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    diff = np.power(np.subtract(P, x.T),2)
    ss   = np.sum(-A*diff, axis=1)
    exp  = np.exp(ss)

    y = -np.dot(alpha, exp)

    if normalize:
        y = 2*(y-minv)/(maxv-minv)-1

    return y

def hartmann3_tf(x):
    x = tf.reshape(x, [-1,1])
    x = (x+1)/2

    minv = -3.86278
    maxv = 18.06

    A = tf.constant([[3.0, 10, 30],[0.1, 10, 35],[3.0, 10, 30],[0.1, 10, 35]], dtype=tf.float32)
    P = 0.0001 * tf.constant([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]], dtype=tf.float32)
    alpha = tf.constant([1.0, 1.2, 3.0, 3.2], dtype=tf.float32, shape= [1,4])

    diff = tf.subtract(P, tf.transpose(x))**2

    ss = tf.reduce_sum( tf.multiply(-A, diff ), 1)
    ss = tf.reshape(ss, [-1,1])
    exp  = tf.exp(ss)

    y = -tf.matmul(alpha, exp)

    y = 2*(y-minv)/(maxv-minv)-1

    return tf.reshape(y, (-1,1))

def parabolasin(x, wiggle=2.0):
	minv = -2.0
	maxv = 5.0

	x = np.array(x)

	y = np.sum(x**2)+np.sum(np.sin(wiggle*x))

	y = 2*(y-minv)/(maxv-minv)-1

	return y

def parabolasin_tf(x, wiggle=2.0):
	minv = -2.0
	maxv = 5.0

	y = tf.reduce_sum(x**2, axis=1)+tf.reduce_sum(tf.sin(wiggle*x),axis = 1)

	y = 2*(y-minv)/(maxv-minv)-1

	return tf.reshape(y, (-1,1))

def styblinski4(x, normalize=True):
	minv = -39.166*4
	maxv = 40.0

	x = np.array(x)*5

	y = np.sum(x**4-16*x**2+5*x)/2.0

	if normalize:
		y = 2*(y-minv)/(maxv-minv)-1

	return y

def styblinski4_tf(x, normalize=True):
	minv = -39.166*4
	maxv = 40.0

	x = x*5

	y = tf.reduce_sum(x**4-16*x**2+5*x,axis=1)/2

	if normalize:
		y = 2*(y-minv)/(maxv-minv)-1

	return tf.reshape(y, (-1,1))



def hartmann6(x, normalize=True):
    x = np.array(x)

    if normalize:
        x = (x+1)/2

    minv = -3.32237
    maxv = 38.7

    A = np.array([[10, 3, 17, 3.5, 1.7, 8],[0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8],\
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],[2329, 4135, 8307, 3736, 1004, 9991],\
                         [2348, 1451, 3522, 2883, 3047, 6650],[4047, 8828, 8732, 5743, 1091, 381]])
    alpha = np.array([1, 1.2, 3, 3.2])

    diff = np.power(np.subtract(P, x.T),2)
    ss   = np.sum(-A*diff, axis=1)
    exp  = np.exp(ss)

    y = -np.dot(alpha, exp)

    if normalize:
        y = 2*(y-minv)/(maxv-minv)-1

    return y

def hartmann6_tf(x, normalize=True):

	x = tf.reshape(x, [-1,1])

	if normalize :
		x = (x+1)/2

	minv = -3.32237
	maxv = 38.7

	A = tf.constant([[10, 3, 17, 3.5, 1.7, 8],[0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8],\
					[17, 8, 0.05, 10, 0.1, 14]], dtype=tf.float32)
	P = 1e-4 * tf.constant([[1312, 1696, 5569, 124, 8283, 5886],[2329, 4135, 8307, 3736, 1004, 9991],\
							[2348, 1451, 3522, 2883, 3047, 6650],[4047, 8828, 8732, 5743, 1091, 381]], dtype=tf.float32)
	alpha = tf.constant([1, 1.2, 3, 3.2], dtype=tf.float32)

	diff = tf.subtract(P, tf.transpose(x))**2

	ss = tf.reduce_sum( tf.multiply(-A, diff ), 1)
	ss = tf.reshape(ss, [-1,1])
	exp  = tf.exp(ss)

	y = -tf.matmul(tf.reshape(alpha,[1,-1]), exp)

	if normalize:
		y = 2*(y-minv)/(maxv-minv)-1

	return tf.reshape(y, (-1,1))
