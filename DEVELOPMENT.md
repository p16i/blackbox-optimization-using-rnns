## Running experiments on a server
### Requirement
    - TensorFlow
    - Install all dependencies by running `pip install requirements.txt`
    - Access to TUB network (VPN)

### Setup project
1. SSH to the server. For Windows, you can do this using Putty.

```
ssh username@server-ip
```

2. Clone the repository and activate `virtual-env` there.

```
git clone git@github.com:heytitle/neural-network-project.git

```

3. Once `virtal-env` activated, install dependencies

```
(neural-network-project) $  pip install -r requirements.txt

```

### Run an experiment
Each experiment is corresponding to training `RNN` on training data determined with `--dimension` with combinations of hyperparameter specifid in `config.yaml`. The `experiment-manager.py` will write training logs to a directory(`--log-dir`) under `LOG_BASE`, for example `./log/2d`. The example below shows how to run the command.

```
TF_CPP_MIN_LOG_LEVEL=3 python experiment-manager.py run --dimension 2 --log_dir 2d --epochs 2
```

**NOTE : Before running any experiment** 

	1. Make sure that the resposity is up-to-dated, otherwise the results might not be consistent
	2. Make sure that `--log_dir` is empty.

### Useful commands
1. `tmux`
	
	If you want to leave the experiment running while you're logging out from the server. You have to use [tmux](https://tmux.github.io/). You can do this by running `tmux` after log-in to the server.
2. `nvidia-smi`
	
	Monitor GPU usage. Combining this command with `watch` to get near realtime updated.
	
	```
	$ watch -n 2 nvidia-smi #update every 2 secs
	```

## Experiment Setting
### 1D
```
no_training: 10000
no_testing : 1000

# hyperparameter
n_hidden: [10,20,30,40,50]
loss_functions: [EI, F_SUM, .. ] # don't need right now
```

### 2D
```
no_training: 20000
no_testing : 2000

# hyperparameter
n_hidden: [30,40,50,100]
loss_functions: [EI, F_SUM, .. ] # don't need right now
```

### 3D
```
no_training: 30000
no_testing : 3000

# hyperparameter
n_hidden: [50, 100, 200]
loss_functions: [EI, F_SUM, .. ] # don't need right now
```

### 4D
```
no_training: 40000
no_testing : 4000

# hyperparameter
n_hidden: [100, 200, 250]
loss_functions: [EI, F_SUM, .. ] # don't need right now
```

### 5D
```
no_training: 50000
no_testing : 5000

# hyperparameter
n_hidden: [200,300,400]
loss_functions: [EI, F_SUM, .. ] # don't need right now
```
