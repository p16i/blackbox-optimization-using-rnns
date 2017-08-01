import skopt
import warnings
import numpy as np
import fire
import utils
import gpfunctions as gp
import time
from scipy.optimize import basinhopping


class SKOptimizer:
    def get_samples_sk(self, X,A,minv,maxv, l, dim, n_steps, function, kernel, n, x_start, optimizer):
        t_start = time.time()

        # performs skopt optimization for the first n gp-functions specified by the parameters X,A,minv,maxv

        # the i-th gp-function
        fun = lambda x, i : np.asscalar(function("np", X[np.newaxis,i], A[np.newaxis,i],\
                                minv[np.newaxis,i], maxv[np.newaxis,i], l, kernel, \
                                np.array(x)[np.newaxis,np.newaxis,:]))

        samples_sk_x = []
        samples_sk_y = []

        opt = {
            "gp": skopt.gp_minimize,
            "forest": skopt.forest_minimize,
            "random": skopt.dummy_minimize,
            "gbrt": skopt.gbrt_minimize,
            "basinhopping": basinhopping
        }[optimizer]

        for i in range(n):

            evalute_gp_func = lambda x: fun(x,i)

            if optimizer in ['gp','forest', 'random', 'gbrt']:
                res = opt( evalute_gp_func, [(-1.0, 1.0)]*dim, n_calls=n_steps, x0=x_start)

                samples_sk_x += [np.array(res.x_iters)]
                samples_sk_y += [np.array(res.func_vals)]
            elif optimizer is 'basinhopping':

                samples_x = [x_start]
                samples_y = [evalute_gp_func(x_start)]

                def callback_func(x, f_x, accepted):
                    samples_x.append(x)
                    samples_y.append(f_x)

                minimizer_kwargs = dict(method='L-BFGS-B', bounds = [(-1,1)]*dim)
                res = opt( evalute_gp_func, x0=x_start, minimizer_kwargs=minimizer_kwargs, niter=n_steps-1, callback= callback_func )

                samples_sk_x += samples_x
                samples_sk_y += samples_y


        print("Time: ",time.time()-t_start)

        #print("shape: ", np.array(samples_sk_y).shape)

        return np.array(samples_sk_x).reshape(n, n_steps, dim), np.array(samples_sk_y).reshape(n, n_steps)

    def run(self, dim, kernel, n_steps=21, no_testing_func=10, optimizer = 'gp', dataset='normal'):
        print("Optimizing for first %d functions of %d-%s testing data using %s optimizer with %d steps" % (no_testing_func, dim, kernel, optimizer, n_steps))
        print('dataset %s' % dataset)
        conf = utils.loadConfig()

        x0 = conf['experiments']["%dD" % dim]['hyperparameters']['starting_point'][0]

        X, A, minv, maxv = utils.loadData(dim, 'testing', kernel)

        n_bumps = 6
        l = 2/n_bumps*np.sqrt(dim)

        dataset_func = gp.dataset_function(dataset)
        kernel_func = gp.kernel_function(kernel)

        samples_x, samples_y = self.get_samples_sk(X, A, minv, maxv, l, dim, n_steps, dataset_func, kernel_func, no_testing_func, x0, optimizer = optimizer)

        base = kernel
        if dataset != 'normal':
            base = '%s-%s'% (kernel, dataset)

        directory = '%s/%sd-%s' % ( optimizer, dim, base )

        print('Saving data with prefix %s' % directory )
        self.save_samples(samples_x, samples_y, directory)

    def save_samples(self, samples_x, samples_y, directory):
        base_dir = utils.loadConfig()['SKOPT_RESULT_BASE_DIR']
        np.save( '%s/%s-samples_x' % (base_dir, directory), samples_x)
        np.save( '%s/%s-samples_y' % (base_dir, directory), samples_y)

if __name__ == "__main__":
    fire.Fire(SKOptimizer)
