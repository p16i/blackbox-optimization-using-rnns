## Setup project

### Requirements
    - TensorFlow
    - XFoil

1. Clone the repository and activate `virtual-env` there.

```
git clone git@github.com:heytitle/blackbox-optimization-using-rnns.git

```

3. Once `virtal-env` activated, install dependencies

```
(neural-network-project) $  pip install -r requirements.txt

```
4. Setup Python's path
```
# in bin/activate (virtualenv's activate)
export PYTHONPATH="./src:$PYTHONPATH"
```

## Train a lstm model

Each experiment is corresponding to training `RNN` on training data determined with `--dimension` with combinations of hyperparameter specifid in `config.yaml`. The `experiment-manager.py` will write training logs to a directory(`--log-dir`) under `LOG_BASE`, for example `./log/2d`. The example below shows how to run the command.

```
TF_CPP_MIN_LOG_LEVEL=3 python ./scripts/train-lstm-model.py run --dimension 2 --log_dir 2d --kernel rbf --epochs 2
```

**NOTE : Before running any experiment** 

	1. Make sure that the resposity is up-to-dated, otherwise the results might not be consistent
	2. Make sure that `--log_dir` is empty.

#### Useful commands
1. `tmux`
	
	If you want to leave the experiment running while you're logging out from the server. You have to use [tmux](https://tmux.github.io/). You can do this by running `tmux` after log-in to the server.
2. `nvidia-smi`
	
	Monitor GPU usage. Combining this command with `watch` to get near realtime updated.
	
	```
	$ watch -n 2 nvidia-smi #update every 2 secs
	```

## Experiments
### SKOpt's optimizers on GP test data
```
python ./scripts/skopt-test-data-experiment.py run --kernel rbf --dim 6 --no_testing_func 2000 --optimizer gp --dataset prior0 --n_steps 21
```

### Airfoil Optimization
This experiment requires [xfoil](http://web.mit.edu/drela/Public/web/xfoil/).
```
# Make sure XFOIL_PATH set properly
python ./scripts/airfoil-experiment.py run --optimizer basinhopping  --normalization 100 --dim 6
```

### MNIST Hyperparameters Optimization
```
python ./scripts/mnist-experiment.py run --optimizer basinhopping  --no_runs 5
```
