import utils
from sklearn.model_selection import ParameterGrid
import time
import rnn_n2n
import fire
import os


class ExperimentManager:
    def run( self, dimension, log_dir ):
        config = utils.loadConfig()
        # todo: check if log dir doesn't exists otherwise fail
        log_location = config.BASE_LOG_DIR  + '/' + log_dir
        print(log_location)
        if os.isdir(log_location):
            die("log dir already exists, please specify new one")

        experiment_conf = config['experiments']['%dD'%dimension]

        hyper_params = experiment_conf['hyperparameters']
        param_grid = list(ParameterGrid(hyper_params))

        total_combinations = len(param_grid)
        print('Running experiments of %dD with %d hyperparameter combinations' % (dimension, total_combinations) )

        start = time.time()
        for i in range(total_combinations):
            params = param_grid[i]
            # todo: create logger file
            print('%3d/%d - %s' % ( i+1, total_combinations, params ))
            rnn_n2n.train_rnn_n2n(dimension, epochs=1, **params)
        end = time.time()

        print("Finished %d combinations using %.4f mins"%( total_combinations, (end-start)/60.0 ))
        print("==========================================")



if __name__ == "__main__":
    # manager = ExperimentManager()
    # manager.run(2)
    fire.Fire(ExperimentManager)
