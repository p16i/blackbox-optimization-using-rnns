import fire
import utils
import numpy as np
from sklearn.model_selection import ParameterGrid
import lstm_model
import airfoil_simulator
import skopt

class OptimizeAirfoil:
    def run(self, optimizer='lstm', dim=2, no_steps=20, loss="MIN", kernel="rbf", normalization=100):
        print('Optimize Airfoil with %s optimizer' % optimizer)

        config = utils.loadConfig()['airfoil_optimization']

        output_dir = config['output_dir']

        alphas = np.linspace( config['alpha_range'][0], config['alpha_range'][1], config['no_alpha'] )

        y_pairs = list(self.generate_pairs(config['adjustable_y']))

        print('Generate foils with %d alphas' % len(alphas))
        print('and %d pairs of y' % len(y_pairs))

        params = {'alpha': alphas, 'y_pair': y_pairs}
        param_grid = ParameterGrid(params)


        total_combination = len(param_grid)
        print('We have %d foils to run.' %  total_combination)

        x_0 = np.array([ config['x_start'] ]*dim).reshape(-1,dim)

        results_x = np.zeros((total_combination, no_steps+1, dim))
        results_y = np.zeros((total_combination, no_steps+1))

        method = optimizer
        for i in range(total_combination):
            param = param_grid[i]
            param['pos1'] = param['y_pair'][0]
            param['pos2'] = param['y_pair'][1]
            del param['y_pair']

            print('Evaluating - %d ' % (i+1) )
            print(param)

            if optimizer is 'lstm':
                print('Using LTSM')
                model = utils.get_trained_model(dim=2, kernel=kernel, loss=loss)
                samples_x, samples_y = self.optimize_lstm(x_0, model, param, no_steps, normalization)

                method = '%s-%s-%s' % (optimizer, loss, kernel)
            else:
                obj_func = lambda x: self.obj_airfoil_lift_drag(x, param, normalization)

                if optimizer is 'random':
                    print('Optimize Randomly')
                    samples_x, samples_y = self.optimize_random(x_0.flatten(), no_steps+1, obj_func)
                    samples_x = np.array(samples_x).flatten()
                elif optimizer is 'skopt':
                    samples_x, samples_y = self.optimize_skopt(x_0.flatten(), no_steps+1, obj_func)


            results_x[i,:,:] = np.array(samples_x).reshape(1,-1,dim)
            results_y[i,:] = samples_y


        print('Saving result to %s' % (method) )
        np.save( '%s/normalize-%d/%s-samples_x' % (output_dir, normalization, method), results_x)
        np.save( '%s/normalize-%d/%s-samples_y' % (output_dir, normalization, method), results_y)


    def optimize_lstm(self, x_0, model, foil_params, steps, normalization):
        sess, model_params = lstm_model.load_trained_model(model)
        samples_x, samples_y = lstm_model.generate_sample_sequence(sess, model_params, x_0, steps = steps, \
            obj_func=lambda x: np.array(self.obj_airfoil_lift_drag(x, foil_params, normalization)).reshape(1,-1) \
        )

        samples_y = np.array(samples_y).flatten()

        return samples_x, samples_y

    def optimize_random(self, x_0, steps, obj_func):
        res = skopt.dummy_minimize(obj_func, dimensions=[(-1.0,1.0)]*2, n_calls=steps, x0=x_0)
        return res.x_iters, res.func_vals

    def optimize_skopt(self, x_0, steps, obj_func):
        res = skopt.gp_minimize(obj_func, dimensions=[(-1.0,1.0)]*2, n_calls=steps, n_random_starts=10, x0=x_0)
        return res.x_iters, res.func_vals

    def obj_airfoil_lift_drag(self, x, foil_params, normalization):
        x = np.array(x)
        obj_value = -1*airfoil_simulator.objective(x.reshape(-1), **foil_params)/normalization
        return obj_value


    def generate_pairs(self, arr):
        total_items = len(arr)
        for i in range(total_items):
            for j in range(1, total_items):
                yield (arr[i],arr[j])


if __name__ == "__main__":
    fire.Fire(OptimizeAirfoil)
