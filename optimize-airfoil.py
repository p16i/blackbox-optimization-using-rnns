import fire
import utils
import numpy as np
from sklearn.model_selection import ParameterGrid
import lstm_model
import airfoil_simulator
import skopt
from scipy.optimize import basinhopping

class OptimizeAirfoil:
    def run(self, optimizer='lstm', dim=2, no_steps=20, loss="MIN", kernel="rbf", normalization=100):
        print('Optimize Airfoil with %s optimizer' % optimizer)

        config = utils.loadConfig()['airfoil_optimization']

        output_dir = config['output_dir']

        alphas = np.linspace( config['alpha_range'][0], config['alpha_range'][1], config['no_alpha'] )

        # y_pairs = list(self.generate_pairs(config['adjustable_y']))
        y_pairs = [
            (3,4),
            (4,5)
        ]

        print('Generate foils with %d alphas' % len(alphas))
        print('and %d pairs of y' % len(y_pairs))

        params = {'alpha': alphas, 'y_pair': y_pairs}
        param_grid = ParameterGrid(params)


        total_combination = len(param_grid)
        # total_combination = 1
        print('We have %d foils to run.' %  total_combination)


        results_x = np.zeros((total_combination, no_steps+1, dim))
        results_y = np.zeros((total_combination, no_steps+1))

        method = optimizer
        for i in range(total_combination):
            param = param_grid[i]
            param['pos1'] = param['y_pair'][0]
            param['pos2'] = param['y_pair'][1]
            del param['y_pair']

            input_space = [
                config['y_input_space'][param['pos1']],
                config['y_input_space'][param['pos2']]
            ]

            print(input_space)

            print('Evaluating - %d ' % (i+1) )
            print(param)

            if optimizer is 'lstm':
                x_0 = np.array([ config['x_start'] ]*dim).reshape(-1,dim)
                print('Using LTSM')
                model = utils.get_trained_model(dim=2, kernel=kernel, loss=loss)
                samples_x, samples_y = self.optimize_lstm(x_0, model, param, no_steps, normalization, input_space)

                method = '%s-%s-%s' % (optimizer, loss, kernel)
            else:
                obj_func = lambda x: self.obj_airfoil_lift_drag(x, param, normalization)

                x_0 = np.array([ x[0] for x in input_space ])
                print('Starting %s' % x_0)

                if optimizer is 'random':
                    print('Optimize Randomly')
                    samples_x, samples_y = self.optimize_random(x_0.flatten(), no_steps+1, obj_func, input_space)
                    samples_x = np.array(samples_x).flatten()
                elif optimizer is 'skopt':
                    samples_x, samples_y = self.optimize_skopt(x_0.flatten(), no_steps+1, obj_func, input_space)
                elif optimizer is 'basinhopping':
                    print('Using basinhopping')
                    samples_x, samples_y = self.optimize_basinhopping(x_0.flatten(), no_steps+1, obj_func, input_space)



            results_x[i,:,:] = np.array(samples_x).reshape(1,-1,dim)
            results_y[i,:] = samples_y


        print('Saving result to %s' % (method) )
        np.save( '%s/normalize-%d/%s-samples_x' % (output_dir, normalization, method), results_x)
        np.save( '%s/normalize-%d/%s-samples_y' % (output_dir, normalization, method), results_y)


    def optimize_lstm(self, x_0, model, foil_params, steps, normalization, input_space):

        original_spaces = np.array([(-1,1)]*len(input_space))
        input_space = np.array(input_space)

        obj_func = lambda x: np.array(self.obj_airfoil_lift_drag( self.input_scaling(x, original_spaces, input_space), foil_params, normalization)).reshape(1,-1)
        sess, model_params = lstm_model.load_trained_model(model)
        samples_x, samples_y = lstm_model.generate_sample_sequence(sess, model_params, x_0, steps = steps, \
            obj_func = obj_func \
        )

        samples_y = np.array(samples_y).flatten()

        return samples_x, samples_y

    def optimize_random(self, x_0, steps, obj_func, input_space):
        res = skopt.dummy_minimize(obj_func, dimensions=input_space, n_calls=steps, x0=x_0)
        return res.x_iters, res.func_vals

    def optimize_skopt(self, x_0, steps, obj_func, input_space):
        res = skopt.gp_minimize(obj_func, dimensions=input_space, n_calls=steps, n_random_starts=10, x0=x_0)
        return res.x_iters, res.func_vals

    def optimize_basinhopping(self, x_0, steps, obj_func, input_space):
        samples_x = [x_0]
        samples_y = [obj_func(x_0)]

        def callback_func(x, f_x, accepted):
            samples_x.append(x)
            samples_y.append(f_x)

        minimizer_kwargs = dict(method='L-BFGS-B', bounds = input_space, options=dict(disp=False, maxiter=1))
        basinhopping( obj_func, x0=x_0, minimizer_kwargs=minimizer_kwargs, niter=steps-1, callback= callback_func )

        return samples_x, samples_y

    def obj_airfoil_lift_drag(self, x, foil_params, normalization):
        x = np.array(x) / 5.0
        obj_value = -1*airfoil_simulator.objective(x.reshape(-1), **foil_params)/normalization
        return obj_value

    def input_scaling(self, x, original_spaces, new_spaces):

        ow = original_spaces[:,1] - original_spaces[:,0]

        nw = new_spaces[:,1] - new_spaces[:,0]
        ratio = nw/ow

        new_x = x*ratio - (original_spaces[:,0]*ratio - new_spaces[:,0])

        return new_x

    def generate_pairs(self, arr):
        total_items = len(arr)
        for i in range(total_items):
            for j in range(1, total_items):
                yield (arr[i],arr[j])


if __name__ == "__main__":
    fire.Fire(OptimizeAirfoil)
