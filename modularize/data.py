import numpy as np
import gpfunctions as gp

def gen_data(n_train, n_test, dim, n_bumps, l, n_mc_samples = 100):
    X = np.random.uniform(low = -1.0, high = 1.0, size = (n_train+n_test, n_bumps, dim))
    Y = np.random.uniform(low = -1.0, high = 1.0, size = (n_train+n_test, n_bumps))

    K_XX = gp.kernel(X,X,l)
    A = np.linalg.solve(K_XX, np.expand_dims(Y,axis=2))
    
    mc_samples = np.random.uniform(low = -1.0, high = 1.0, size = [1, n_mc_samples, dim])
    y = gp.GP(X,A,mc_samples,l)

    min_vals = np.min(y, axis = 1).reshape(n_train+n_test,1)
    max_vals = np.max(y, axis = 1).reshape(n_train+n_test,1)

    return (X[:n_train], A[:n_train], min_vals[:n_train], max_vals[:n_train],\
            X[-n_test:], A[-n_test:], min_vals[-n_test:], max_vals[-n_test:])