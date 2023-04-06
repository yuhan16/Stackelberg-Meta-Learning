# sg-meta 
This repo is for Stackelberg meta-learning project. The underlying application is UAV guiding UGV.


## File Specifications
The following scripts are used for different purposes:
- `param.py`: definitions of problem parameters.
- `utils.py`: definitions of utility functions for executing the algorithm.
- `main.py`: main script for sg meta-learning algorithms and implementations for receding horizon planning and other compared algorithms.
- `main_ave_br`: script to implement parameter-ave learning.
- `main_ave_largenn:`: script to implement output-ave learning.
- `main_no_guide`: script to implement zero guidance control.
- `misc_test.py`: test scripts for different function testings.
- `plot_figs.py`: functions to plot figures.
- `data/`: data directory. Store necessary data including plots and key training data for reference.
- `requirements.txt`: required python packeges for the project. See [Installation Instructions](#installation-instructions).


## Installation Instructions
We use `Pytorch` to train the neural network model in our approach.

First, create a virtual environment with python virtulenv module.
```bash
$ python3 -m venv <venv-name>
```

Second, enter the created virtual environment and install required python packages using pip and provided requirements.txt:
```bash
$ source /path_to_venv-name/bin/activate    # enter virtual environment 
(venv-name)$ pip install -r requirements.txt
```

Then you are ready to run all the scripts.


## Coding specifications

To save space, we use `state_dict` to record a neural network.

### obstacle
Each obstacle is specified by a 6-dim vector: `[xc, yc, rc, norm, x_scale, y_scale]`.
- If `norm=1`, `x_scale, y_scale` scale the unit width/height `rc`.
- If `norm=2`, `x_scale, y_scale` scale the radius `rc`.
- If `norm=-1`, `x_scale, y_scale` scale the unit edge length `rc`.
The previous scaling notation is easy for plotting. The math representation is scaled by `1/x_scale` and `1/y_scale` respectively.

### BR data
- BR data are organized into numpy array. `D[i,:] = [x, a, br]`
- Trajectory is stored in a 2d numpy array with axis0 as time index. `x_traj[t, :] = x_t`
- Use a list to store type-related quantity. `br_list[i]` is the adapted meta model (an NN) for type i follower.
  - Trajectories have different time dimension.
  - state trajectory `x_traj` has dimension T+1. `x_0, ..., x_T`
  - control input trajectories `a_traj` and `b_traj` have dimension T. `a_0, ..., a_{T-1}`
  - costate trajectory `lam_traj` has dimension T. `lam_1, ..., lam_T`


### Utilities
- Use one global parameter file, which defines type-related information.
- Five classes to implement different functions: `Leader`, `Follower`, `BRNet`, `Meta`, `PlotUtils`.
  - `Leader`: compute leader's objective; solve leader's decision problem.
  - `Follower`: compute follower's objective; compute follower's best response.
  - `BRNet`: definition of the best response NN.
  - `Meta`: sampling and meta-learning algorithm.
  - `PlotUtils`: plotting functions.

In `Leader` class, we specify some functions:
- `compute_opt_traj`: solve the parameterized trajectory optimization problem
  - `initx`: generate initial guess for trajectory optimization problem
  - `oc_opt`: use optimization solver to obtain the trajectory
  - `pmp_opt`: use pmp conditions to refine the trajectory
- `obj_oc`: objective of control cost
- `grad_obj_oc`: gradient of control cost

In `Meta` class, we specify some functions:
- sample_task_theta: sample BR data for task theta
- sample_task_theta_traj: sample BR data for task theta near the trajectory
- sample_task_theta_uniform: randomly sample BR data for task theta
- update_model: update meta model
- update_model_theta: update intermediate model
- train_brnet: train separate brnet for different followers, designed for individual learning


### Checklist for small tests
- [x] NN output and jac computation
- [x] leader's obs jac, dynamics jac
- [x] leader opt
- [x] leader pmp
- [x] leader's initial trajectory
- [x] follower br opt, jac
- [x] meta-learning inner GD update
- [x] meta-learning outer GD update
- [x] data sampling, random and trajectory
- [x] sg adaptation