# Stackelberg Meta-Learning for Cooperative Trajectory Guidance
This repo is for Stackelberg meta-learning project. The underlying application is UAV guiding UGV.


## File Structure Specifications
- `data/`: data directory. Store necessary data including plots and key training data for reference.
- `logs/`: log directory. Stroe training log for diagnosis.
- `bash/`: bash scripts to run the algorithm in order to generate log files (redirect output to `logs/`)
- `requirements.txt`: required python packeges for the project. Installed with `pip`.
- Python scripts.

### Python Scripts
The following scripts are used for different purposes:
- `param.py`: definition of problem parameters.
- `utils_meta.py`: definition of utility functions for executing the algorithm.
- `main_meta`: main script for sg meta-learning algorithms and implementations for other compared algorithms.
- `misc_test`: test scripts for different function testing.
- `plot_figs.py`: functions to plot figures.

### data directory
We have following sub directories in `data/`:
- `plots/`: stores simulation plots.
- `br_data/`: stores pre-sampled uniform BR data for all scenarios.
  - `br_data/scenario(x)/`: strores pre-sampled BR data for scenario x.
- `data_meta/`: stores meta learning related data.
  - `data_meta/scenario(x)/`: stores related data for scenario x, including adapted BR model.
- `data_nometa`: stores no meta learning related data.
- `data_noguide`: stores no guidance related data.

### log directory
The log directory has the same structure as `data/`.


## Coding Specifications
- BR data are organized into numpy array. `D[i,:] = [x, a, br]`
- Trajectory is stored in a 2d numpy array with axis0 as time index. `x_traj[t, :] = x_t`
- Use a list to store type-related quantity. `br_list[i]` is the adapted meta model (an NN) for type `i` follower.
- Trajectories have different time dimension. 
  - state trajectory `x_traj` has dimension `T+1`. `x_0, ..., x_T`
  - control input trajectories `a_traj` and `b_traj` have dimension `T`. `a_0, ..., a_{T-1}`
  - costate trajectory `lam_traj` has dimension `T`. `lam_1, ..., lam_T`
- Use `brnet` to store parameter value only. Its gradient `.grad` is useless.


### Class Specifications
- Use one global parameter file, which defines type-related information.
- Five classes to implement different functions: `Leader`, `Follower`, `BRNet`, `Meta`, `Auxiliary`.
  - `Leader`: compute leader's objective; solve leader's decision problem.
  - `Follower`: compute follower's objective; compute follower's best response.
  - `BRNet`: defines the best response NN.
  - `Meta`: sample and algorithm.
  - `Auxiliary`: auxiliary functions such as constraint check and simple plot functions for test.

In `Leader` class, we specify some function structures:
- `solve_oc`: solve the trajectory optimization
  - `opt_solver`: use optimization solver to obtain the trajectory
  - `pmp_solver`: use pmp conditions to refine the trajectory
- `obj_oc`: objective of control cost, same applies to `obj_br`, `obj_L`
- `grad_obj_oc`: gradient of control cost, same applies to `grad_obj_br`, `grad_obj_L`

In `Meta` class, we specify some function structures:
- `sample_task_theta`: sample BR data for task theta
  - `sample_task_theta_traj`: sample BR data for task theta near the trajectory
  - `sample_task_theta_uniform`: randomly sample BR data for task theta
- `update_model`: update meta model
- `update_model_theta`: update intermediate model
- `train_brnet`: train separate brnet for different followers, designed for individual learning


## Installation Instructions
We use [Pytorch](https://pytorch.org/) to train the neural network model in our framework. 

First, create a virtual environment with python virtulenv module.
```bash
$ python -m venv <venv-name>
```

Second, enter the created virtual environment and install required python packages using `pip` and provided `requirements.txt`:
```bash
$ source /path_to_venv-name/bin/activate    # enter virtual environment 
(venv-name)$ pip install -r requirements.txt
```

Then you are ready to run all the scripts.

## Running Bash Scripts
We provide two bash scripts to run the algorithm and redirect the output to `logs/` directory.
- `sg_meta.sh`: for Stackelberg meta-learning
- `non_meta.sh`: for individual learning
If there is no need to generate logs, ignore the bash scripts.

Note that since we encapsulate different learning algorithms into separate functions in `main_test.py`, we need to **comment unrelated functions when running the bash scripts**. For example, 
- When running `sg_meta.sh`, only leave function `sg_meta_learn()` in `main_test.py` uncommented.
- When running `non_meta.sh`, only leave function `individual_learn()` in `main_test.py` uncommented.

We can specify the scenario index in the bash script to run the algorithm for different scenarios.
