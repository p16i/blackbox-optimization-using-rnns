import fire
import tensorflow as tf
import numpy as np

import utils
import gpfunctions as gp
import lstm_model
import benchmarkfunctions as bm
import skopt
import itertools

FUNCTIONS = {
    '2d': bm.parabolasin,
    '2d_tf': bm.parabolasin_tf,
    '3d': bm.hartmann3,
    '3d_tf': bm.hartmann3_tf,
    '4d': bm.styblinski4,
    '4d_tf': bm.styblinski4_tf,
    '6d': bm.hartmann6,
    '6d_tf': bm.hartmann6_tf
}

SEED = 117

class BenchmarkFunctionsExperimentManger:
    def run( self, dim, optimizer, no_transformation_x=10, no_transformation_y=10, n_steps=20, loss='MIN', kernel='rbf' ):

        print('Running experiments on %d-d benchmark function' % dim )

        total_transformations = no_transformation_x * no_transformation_y
        print('with %d transformations' % (total_transformations) )
        print('using %s optimizer' % optimizer)

        np.random.seed(SEED) # this make the 2 command below deterministic
        factor = 0.01
        transformation_x = factor*np.random.uniform(-1,1, (no_transformation_x,dim))
        transformation_y = factor*np.random.uniform(0,1, (no_transformation_y))
        transformation_x[0,:] = 0
        transformation_y[0] = 0

        starting_point = utils.loadConfig()['experiments']['%dD'%dim]['hyperparameters']['starting_point'][0]
        x0 = np.array(starting_point)

        permutedIndices = list(itertools.permutations(range(dim), dim))[:100]
        data = np.zeros((total_transformations*len(permutedIndices), n_steps+1))
        count = 0
        for i in range(transformation_x.shape[0]):
            for j in range(transformation_y.shape[0]):
                trans_x = transformation_x[i,:]
                trans_y = transformation_y[j]

                print('%3d - adjust x %s \t adjust y %s' % (count+1, trans_x, trans_y) )

                for k in range(len(permutedIndices)):

                    indices = np.array(permutedIndices[k])
                    print('   %3d : permuted Indices: %s' % (k+1,str(indices)))
                    x0 = np.array(starting_point)[indices]

                    if optimizer == 'skopt-gp':
                        data[count,:] = self.get_y_from_skopt(dim, x0, n_steps, trans_x, trans_y, indices=indices)
                    elif optimizer == 'random':
                        data[count,:] = self.get_y_from_random(dim, x0, n_steps, trans_x, trans_y, indices=indices)
                    elif optimizer == 'lstm':
                        data[count,:] = self.get_y_from_lstm(dim, loss, kernel, x0, n_steps = n_steps, trans_x = trans_x, trans_y = trans_y, indices=indices)
                    else:
                        raise Exception('no %s optimizer' % optimizer)

                    data[i,:] = data[i,:] - trans_y
                    count += 1

        if optimizer == 'lstm':
            optimizer = '%s-%s-%s' % (optimizer, loss, kernel)

        base_dir = utils.loadConfig()['BENCHMARK_RESULT_BASE_DIR']

        np.save('%s/%dd-%s' % (base_dir, dim, optimizer), data)
        # save to files
        # each optimizer when save 100x21 steps in
        # pass

    def get_y_from_skopt(self, dim, x0, n_steps, trans_x, trans_y, indices,  n_start_random=10):
        obj_func = self._build_function('%dd'%(dim), trans_x, trans_y, indices)

        res = skopt.gp_minimize(lambda x: obj_func(x), [(-1.0, 1.0)]*dim, n_calls=n_steps+1, x0=x0)
        return res.func_vals

    def get_y_from_random(self, dim, x0, n_steps, trans_x, trans_y, indices):
        x = np.random.uniform(-1,1,(dim,n_steps+1))
        x[:, 0] = x0

        func = self._build_function('%dd'%(dim), trans_x, trans_y, indices)
        y = np.apply_along_axis(lambda x: func(x), 0, x)
        return y.T.reshape(-1)

    def get_y_from_lstm(self, dim, loss, kernel, x0, trans_x, trans_y, indices, n_steps=20, debug=False):

        func = self._build_function_tf('%dd_tf'%(dim), trans_x, trans_y, indices)

        model = utils.get_trained_model(dim=dim, kernel=kernel, loss=loss)
        starting_point = x0

        model_params = lstm_model.load_model_params(model, debug=False)

        with tf.Session() as sess:

            lstm_params = {
                'dim' : model_params['dim'],
                'n_hidden': model_params['n_hidden'],
                'forget_bias': model_params['forget_bias'],
                'scope': model_params['scope']
            }
            cell, weights = lstm_model.get_lstm_weights(**lstm_params)

            saver = tf.train.Saver()
            saver.restore(sess, model_params['model_path'])

            benchmark_samples_params = {
                'f': func,
                'cell': cell,
                'weights': weights,
                'dim': model_params['dim'],
                'n_hidden': model_params['n_hidden'],
                'n_steps': model_params['n_steps'],
                'scope': model_params['scope'],
                'batch_size': 1
            }

            samples_benchmark_x, samples_benchmark_y, x_0 = lstm_model.apply_lstm_model(**benchmark_samples_params)

            feed_dict = {
                x_0: np.array(starting_point).reshape(1,-1)
            }
            sample_x, sample_y = sess.run([samples_benchmark_x, samples_benchmark_y], feed_dict=feed_dict)
            sample_x = np.array(sample_x).reshape(-1,dim)[:, indices]
            sample_y = np.array(sample_y).reshape(-1,1).T
            # print(sample_x)
            # print(sample_y)
            # print('------')
            return sample_y.T.reshape(-1)

    def _build_function(self, func_name, trans_x, trans_y, indices):
        return lambda x: FUNCTIONS[func_name](np.array(x)[indices]+trans_x) + trans_y
    def _build_function_tf(self, func_name, trans_x, trans_y, indices):
        return lambda x: FUNCTIONS[func_name](self._tf_swap_indices(x, indices)+trans_x) + trans_y

    def _tf_swap_indices(self, data, indices):
        # print(data.shape)
        # print(indices)
        # # index = tf.Variable([[0,1],[0,0]], dtype=tf.int32)
        # print(tf.gather(tf.transpose(data), indices))
        # print('-----')
        new_x  = tf.gather(tf.transpose(data), indices)
        return tf.transpose(new_x)


if __name__ == "__main__":
    fire.Fire(BenchmarkFunctionsExperimentManger)
