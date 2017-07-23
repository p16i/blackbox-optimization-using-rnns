import fire

import skopt
import utils
import lstm_model
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn import datasets, svm, preprocessing
from sklearn.model_selection import train_test_split

from scipy.optimize import basinhopping
from sklearn.cross_validation import StratifiedKFold

# http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
# GAMMA_MIN = 2**(-15)
# GAMMA_MAX = 2**(2)
# C_MIN     = 2**(-5)
# C_MAX     = 2**(4)
# GAMMA_MIN = 2**(-15)
# GAMMA_MAX = 2**(2)
# C_MIN     = 2**(-5)
# C_MAX     = 2**(10)
GAMMA_MIN = -10
GAMMA_MAX = 4
C_MIN     = -5
C_MAX     = 10

DIMENSIONS = [(GAMMA_MIN, GAMMA_MAX), (C_MIN, C_MAX)]
# DIMENSIONS = [(2**GAMMA_MIN, 2**GAMMA_MAX), (2**C_MIN, 2**C_MAX)]

class MNISTTask:

    def __init__(self):
        self.scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        self.scaler.fit([[GAMMA_MIN, C_MIN], [GAMMA_MAX, C_MAX]])

    def train_and_evaluate(self, gamma=0.1, C=1, seed=200, no_data=5000, no_test_data=1000, debug=False, fold=3):

        mnist = fetch_mldata('MNIST original', data_home='./data')
        # np.random.seed(seed)

        # X_train, X_test = data[indices[:, 0],:], data[indices[:, 1][:no_test_data],:]
        # y_train, y_test = target[indices[:, 0]], target[indices[:, 1][:no_test_data]]
        # indices = np.random.choice( int(mnist.data.shape[0]), (no_data, 2), replace=False)

        print('gamma=%f , C=%f, seed=%d' % (gamma, C, seed))

        data, _, target, _ = train_test_split(mnist.data, mnist.target, test_size=0.96, random_state=seed)

        acc_basket = []

        kf = StratifiedKFold(target, fold)
        for train_index, test_index in kf:

            X_train = data[train_index,:]
            mean, std = np.mean(X_train), np.std(X_train)

            normalize = lambda x : (x - mean)/std

            X_train = normalize(X_train)
            X_test  = normalize(data[test_index,:])

            y_train = target[train_index]
            y_test  = target[test_index]


            classifier = svm.SVC(gamma=gamma, C=C, kernel="rbf")
            classifier.fit(X_train, y_train)

            y_predicted = classifier.predict(X_test)

            acc = np.mean(y_predicted == y_test)
            acc_basket.append(acc)


        acc = np.mean(acc_basket)

        # digits = datasets.load_digits()


        # data   = digits.data
        # target = digits.target

        # X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=seed)


        # classifier = svm.SVC(gamma=gamma, C=C, kernel="rbf")
        # classifier.fit(X_train, y_train)

        # y_predicted = classifier.predict(X_test)

        # acc = np.mean(y_predicted == y_test)

        if debug:
            print("------------")
            print("C: %f" % C)
            print("Gamma: %f" % gamma)
            print("Seed: %f" % seed)
            print("Accuracy: %f" % acc )

        return acc

        # return cost function
    def run(self, optimizer, no_steps=20, loss="MIN", kernel="rbf", no_runs=5):

        np.random.seed(20)
        seeds = np.random.randint(0,10000,(no_runs))

        output_dir = utils.loadConfig()['MNIST_RESULT_BASE_DIR']

        total_seeds = no_runs
        print('Running for %s runs with %s' % (total_seeds, optimizer) )

        results_x = np.zeros((total_seeds, no_steps+1, 2))
        results_y = np.zeros((total_seeds, no_steps+1))


        method = optimizer
        for i in range(total_seeds):
            s = seeds[i]
            print('Seed %d' % s)
            if optimizer is 'lstm':
                x0 = np.array([-1,-1]).reshape(-1,2)
                obj_func = lambda x: self.objective_function(x, s, scaling=True).reshape(1,-1)
                model = utils.get_trained_model(dim=2, kernel=kernel, loss=loss)
                samples_x, samples_y = self.optimize_lstm(x0, model, obj_func, no_steps)

                samples_x = np.array(samples_x).flatten().reshape(-1,2)
                samples_x = self.scaler.inverse_transform(samples_x)

                samples_x = np.power(2, samples_x)
                samples_y = (samples_y-1)/2
                method = '%s-%s-%s' % (optimizer, loss, kernel)
            else:
                x0 = np.array([DIMENSIONS[0][0], DIMENSIONS[1][0]], dtype=np.float)
                obj_func = lambda x: self.objective_function(x, s)
                if optimizer is 'random':
                    print('Optimize Randomly')
                    samples_x, samples_y = self.optimize_random(x0, no_steps+1, obj_func)
                elif optimizer is 'gp':
                    samples_x, samples_y = self.optimize_gp(x0, no_steps+1, obj_func)
                elif optimizer is 'basinhopping':
                    samples_x, samples_y = self.optimize_basinhopping(x0, no_steps+1, obj_func)

                samples_x = np.array(samples_x).flatten().reshape(-1,2)


            results_x[i,:,:] = np.array(samples_x).reshape(1,-1,2)
            results_y[i,:] = samples_y

        print('Saving result to %s' % (method) )
        np.save( '%s/%s-samples_x' % (output_dir, method), results_x)
        np.save( '%s/%s-samples_y' % (output_dir, method), results_y)
    def objective_function(self, x, seed, scaling=False):
        if scaling:
            x = self.scaler.inverse_transform(x)[0]
        x = np.power(2.0, x)

        acc = self.train_and_evaluate(x[0], x[1], seed)
        print('acc  > %f' % acc)

        if scaling:
            acc = -2*acc + 1
        else:
            acc = -acc
        return acc

    def optimize_random(self, x0, steps, obj_func):
        res = skopt.dummy_minimize(obj_func, dimensions=DIMENSIONS, n_calls=steps, x0=x0)
        return res.x_iters, res.func_vals

    def optimize_gp(self, x0, steps, obj_func):
        res = skopt.gp_minimize(obj_func, dimensions=DIMENSIONS, n_calls=steps, n_random_starts=5, x0=x0)
        return res.x_iters, res.func_vals

    def optimize_lstm(self, x0, model, obj_func, steps):

        sess, model_params = lstm_model.load_trained_model(model)
        samples_x, samples_y = lstm_model.generate_sample_sequence(sess, model_params, x0, steps = steps, obj_func = obj_func)

        samples_y = np.array(samples_y).flatten()

        return samples_x, samples_y

    def optimize_basinhopping(self, x0, steps, obj_func):
        samples_x = [x0]
        samples_y = [obj_func(x0)]

        def callback_func(x, f_x, accepted):
            samples_x.append(x)
            samples_y.append(f_x)

        minimizer_kwargs = dict(method='L-BFGS-B', bounds = DIMENSIONS, options=dict(disp=False, maxiter=1))
        basinhopping( obj_func, x0=x0, minimizer_kwargs=minimizer_kwargs, niter=steps-1, callback= callback_func )

        return samples_x, samples_y

if __name__ == "__main__":
    fire.Fire(MNISTTask)

