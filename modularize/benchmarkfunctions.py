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

def goldstein_price(x):
    x = x*2
    
    minv = 3
    maxv = 1020000 
    x_1, x_2 = np.split(x,[1],axis=-1)
	
    y = (1 + (x_1 + x_2 + 1)**2 * (19 - 14*x_1 + 3*x_1**2 - 14*x_2 + 6*x_1*x_2+ 3*x_2**2))*\
    (30 + (2*x_1-3*x_2)**2 * (18 - 32*x_1 + 12*x_1**2 + 48*x_2 - 36*x_1*x_2+ 27*x_2**2))
    
    y = 2*(y-minv)/(maxv-minv)-1
    
    return y
	
def goldstein_price_tf(x):
    x = x*2
    
    minv = 3
    maxv = 1020000
    
    y = (1 + (x[:,0] + x[:,1] + 1)**2 * (19 - 14*x[:,0] + 3*x[:,0]**2 - 14*x[:,1] + 6*x[:,0]*x[:,1]\
                                             + 3*x[:,1]**2))*\
    (30 + (2*x[:,0]-3*x[:,1])**2 * (18 - 32*x[:,0] + 12*x[:,0]**2 + 48*x[:,1] - 36*x[:,0]*x[:,1]\
                                        + 27*x[:,1]**2))
    
    y = 2*(y-minv)/(maxv-minv)-1
    
    return tf.reshape(y, (-1,1))

def hartmann3(x):
    x = (x+1)/2
    
    minv = -3.86278
    maxv = 18.06
    
    A = np.array([[3.0, 10, 30],[0.1, 10, 35],[3.0, 10, 30],[0.1, 10, 35]])
    P = 0.0001 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    
    y = np.sum(alpha[:,np.newaxis] * np.exp(-A*((x[:,:,np.newaxis,:]-P)**2)),axis=(-1,-2))
    
    y = 2*(y-minv)/(maxv-minv)-1
    
    return y

def hartmann3_tf(x):
    x = (x+1)/2
    x = tf.expand_dims(x,0)
    
    minv = -3.86278
    maxv = 18.06
    
    A = tf.constant([[3.0, 10, 30],[0.1, 10, 35],[3.0, 10, 30],[0.1, 10, 35]], dtype=tf.float32)
    P = 0.0001 * tf.constant([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]], dtype=tf.float32)
    alpha = tf.constant([1.0, 1.2, 3.0, 3.2], dtype=tf.float32)
    
    y = tf.reduce_sum(tf.expand_dims(alpha,1) * tf.exp(-A*((tf.expand_dims(x,2)-P)**2)),axis=(-1,-2))
    
    y = 2*(y-minv)/(maxv-minv)-1
    
    return tf.reshape(y, (-1,1))
    
def hartmann6(x):
    x = (x+1)/2
    
    minv = -3.32237
    maxv = 38.7
    
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],[0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8],\
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],[2329, 4135, 8307, 3736, 1004, 9991],\
                         [2348, 1451, 3522, 2883, 3047, 6650],[4047, 8828, 8732, 5743, 1091, 381]])
    alpha = np.array([1, 1.2, 3, 3.2])
    
    y = np.sum(alpha[:,np.newaxis] * np.exp(-A*((x[:,:,np.newaxis,:]-P)**2)),axis=(-1,-2))
    
    y = 2*(y-minv)/(maxv-minv)-1
    
    return y

def hartmann6_tf(x):
    x = (x+1)/2
    x = tf.expand_dims(x,0)
    
    minv = -3.32237
    maxv = 38.7
    
    A = tf.constant([[10, 3, 17, 3.5, 1.7, 8],[0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8],\
                  [17, 8, 0.05, 10, 0.1, 14]], dtype=tf.float32)
    P = 1e-4 * tf.constant([[1312, 1696, 5569, 124, 8283, 5886],[2329, 4135, 8307, 3736, 1004, 9991],\
                         [2348, 1451, 3522, 2883, 3047, 6650],[4047, 8828, 8732, 5743, 1091, 381]], dtype=tf.float32)
    alpha = tf.constant([1, 1.2, 3, 3.2], dtype=tf.float32)
    
    y = tf.reduce_sum(tf.expand_dims(alpha,1) * tf.exp(-A*((tf.expand_dims(x,2)-P)**2)),axis=(-1,-2))
    
    y = 2*(y-minv)/(maxv-minv)-1
    
    return tf.reshape(y, (-1,1))