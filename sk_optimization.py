import skopt
import warnings
import numpy as np
import fire
import utils
import gpfunctions as gp

class SKOptimizer:
    def get_samples_sk(self, X,A,minv,maxv, l, dim, n_steps, function, kernel, n, x_start, optimizer):
        # performs skopt optimization for the first n gp-functions specified by the parameters X,A,minv,maxv

        # the i-th gp-function
        fun = lambda x, i : np.asscalar(function("np", X[np.newaxis,i], A[np.newaxis,i],\
                                minv[np.newaxis,i], maxv[np.newaxis,i], l, kernel, \
                                np.array(x)[np.newaxis,np.newaxis,:]))

        samples_sk_x = []
        samples_sk_y = []
		
        opt = {"gp": skopt.gp_minimize,"forest": skopt.forest_minimize, 
		"random": skopt.dummy_minimize, "gbrt": skopt.gbrt_minimize}[optimizer]
		
        for i in range(n):

            res = opt(lambda x: fun(x,i), [(-1.0, 1.0)]*dim, n_calls=n_steps, x0=x_start)

            samples_sk_x += [np.array(res.x_iters)]
            samples_sk_y += [np.array(res.func_vals)]

        return np.array(samples_sk_x).reshape(n, n_steps, dim), np.array(samples_sk_y).reshape(n, n_steps)

    def run(self, dim, kernel, n_steps=21, no_testing_func=10):
        print("Optimizing for first %d functions of %d-%s testing data" % (no_testing_func, dim, kernel))
        conf = utils.loadConfig()

        x0 = conf['experiments']["%dD" % dim]['hyperparameters']['starting_point'][0]

        X, A, minv, maxv = utils.loadData(dim, 'testing', kernel)

        n_steps = 21
        n_bumps = 6
        l = 2/n_bumps*np.sqrt(dim)

        kernel_func = gp.kernel_function(kernel)
        samples_x, samples_y = self.get_samples_sk(X, A, minv, maxv, l, dim, n_steps, gp.normalized_gp_function, kernel_func, no_testing_func, x0)

        directory = '%sd-%s' % ( dim, kernel )

        print('Saving data with prefix %s' % directory )
        self.save_samples(samples_x, samples_y, directory)

    def save_samples(self, samples_x, samples_y, directory):
        base_dir = utils.loadConfig()['SKOPT_RESULT_BASE_DIR']
        np.save( '%s/%s-samples_x' % (base_dir, directory), samples_x)
        np.save( '%s/%s-samples_y' % (base_dir, directory), samples_y)

if __name__ == "__main__":
    fire.Fire(SKOptimizer)
