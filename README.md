# sg-meta
This repo is for Stackelberg meta-learning project. The underlying application is UAV guiding UGV.


## File Structure Specifications
- `data/`: data directory. Store necessary data including plots and key training data for reference.
- `logs/`: log directory. Stroe training log for diagnosis.
- `requirements.txt`: required python packeges for the project. Installed with `pip`.
- Python scripts.

### Python Scripts
The following scripts are used for different purposes:
- `param_lqg.py`: definition of problem parameters.
- `utils_lqg.py`: definition of utility functions for executing the algorithm.
- `main_meta`: main script for sg meta-learning algorithms.
- `misc_test`: test scripts for different function testing.
- `plot_fig.py`: functions to plot figures.

(xxx)
- `test_gendata`: generates and stores uniformly sampled BR data in `data_meta`
- `test_mpc`: run MPC and online BR updating; stores data in `data_mpc`
- `test_nometa`: run online BR updating with open loop control; stores data in `data_nometa`
- `test_noguide`: run no guidance (myopic planning); stores data in `data_noguide`

### data directory
We have following sub directories in `data/`:
- `plots/`: stores simulation plots.
- `data_meta/`: stores meta learning related as well as pre-sampled uniform BR data
  - `data_meta/scenario(x)/`: stores related data for scenario x, including pre-sampled BR data and adapted BR model.
- `data_nometa`: stores no meta learning related data.
- `data_noguide`: stores no guidance related data.
- `data_mpc`: stores mpc related data.

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


### Function Specifications
- Use one global parameter file, which defines type-related information.
- Four classes to implement different functions: `Leader`, `Follower`, `BRNet`, `Meta`.
  - `Leader`: compute leader's objective; solve leader's decision problem.
  - `Follower`: compute follower's objective; compute follower's best response.
  - `BRNet`: defines the best response NN.
  - `Meta`: sample and algorithm.

Leader's obj parameter `alp`. Follower's obj parameter `beta`.

Meta:
- `sample_task_theta`
  - `sample_task_theta_traj`
  - `sample_task_theta_uniform`
- `update_model`
  - `grad_obj_L`
  - `one_step_sgd_momentum`
- `update_model_theta`
  - `grad_obj_L`
  - `one_step_sgd_momentum`

Leader:
- `solve_oc`
  - `xxx`


## Installation Instructions
We use [Pytorch](https://pytorch.org/) to train the neural network model in our framework. 

First, create a virtual environment with python virtulenv module.
```bash
$ python -m venv <venv-name>
```

Second, enter the created virtual environment and install required python packages using `pip` and provided `requirements.txt`:
```bash
$ source /path_to_venv-name/bin/activate    # virtual environment 
(venv-name)$ pip install -r requirements.txt
```

Then you are ready to run all the scripts.