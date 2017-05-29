import numpy as np

def branin(x):
    a = 1
    b = 5.1 / (4*np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)
    
    x[:,0] = 7.5*x[:,0]+2.5
    x[:,1] = 7.5*x[:,1]+7.5
    
    minv = 0.397887
    maxv = 307
    
    y = a*(x[:,1] - b*x[:,0]**2 + c*x[:,0] - r)**2 + s*(1-t)*np.cos(x[:,0]) + s
    y = 2*(y-minv)/(maxv-minv)-1
    return y.reshape(-1,1)

def goldstein_price(x):
    x = x*2
    
    minv = 3
    maxv = 1020000
    
    y = (1 + (x[:,0] + x[:,1] + 1)**2 * (19 - 14*x[:,0] + 3*x[:,0]**2 - 14*x[:,1] + 6*x[:,0]*x[:,1] + 3*x[:,1]**2))*\
    (30 + (2*x[:,0]-3*x[:,1])**2 * (18 - 32*x[:,0] + 12*x[:,0]**2 + 48*x[:,1] - 36*x[:,0]*x[:,1] + 27*x[:,1]**2))
    
    y = 2*(y-minv)/(maxv-minv)-1
    
    return y.reshape(-1,1)

def hartmann3(x):
    A = np.array([[3.0, 10, 30],[0.1, 10, 35],[3.0, 10, 30],[0.1, 10, 35]])
    P = 0.0001 * np.array([[3689, 1170, 2673], 4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828])
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    
def hartmann4(x):
    return None