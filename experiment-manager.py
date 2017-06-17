import utils
from sklearn.model_selection import ParameterGrid
import time
import lstm_model
import fire
import os
import tensorflow as tf

class ExperimentManager:
    def run( self, dimension, log_dir, epochs):
        # todo: check if log dir doesn't exists otherwise fail
        config = utils.loadConfig()
        log_location = config['BASE_LOG_DIR']  + '/' + log_dir
        model_location = config['MODEL_DIR'];

        assert not os.path.isdir(log_location), "%s already exists, please specify new one" % log_location

        os.makedirs(log_location)

        experiment_conf = config['experiments']['%dD'%dimension]

        hyper_params = experiment_conf['hyperparameters']
        param_grid = list(ParameterGrid(hyper_params))

        total_combinations = len(param_grid)
        print('Running experiments of %dD with %d hyperparameter combinations' % (dimension, total_combinations) )
        print('log output to %s' % log_location)
        print('save model to %s' % model_location)

        start = time.time()
        for i in range(total_combinations):
            params = param_grid[i]
            filename = str(int(time.time()))
            filepath = log_location + '/' + filename
            f = open(filepath, 'w')

            print('%3d/%d - [log-id: %s] %s ' % ( i+1, total_combinations, filename, params ))

            start_inner = time.time()
            lstm_model.train(dimension, epochs = epochs, logger = f, save_model_path=model_location, **params)
            tf.reset_default_graph()
            end_inner = time.time()

            print('>> took %.4f mins' % ((end_inner - start_inner)/60))

        end = time.time()

        print("Finished %d combinations using %.4f mins"%( total_combinations, (end-start)/60.0 ))
        print("==========================================")



if __name__ == "__main__":
    fire.Fire(ExperimentManager)
