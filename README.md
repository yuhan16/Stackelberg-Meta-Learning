# Stackelberg Meta-Learning for Cooperative Trajectory Guidance
This repo is for Stackelberg meta-learning project. The underlying application is UAV guiding UGV.

**Note:** This repo is reorganized for better readability in April 2024. The old version is archived in the `main-old` branch.

## Requirements
- Python 3.9 or higher
- PyTorch 1.12.1 or higher


## Running Scripts
1. Create a python virtual envrionment with Python 3.9 or higher and source the virtual environment: 
```bash
$ python3,9 -m venv <your-virtual-env-name>
$ source /path-to-venv/bin/activate
```
2. Use `pip` to install related packages:
```bash
(your-venv)$ pip install -e .
```
To use plotting functions, install with
```bash
(your-venv)$ pip install -e ".[visual]"
```
3. Go to the `experiments/` directory and run different training scripts. e.g.,
```bash
(your-venv)$ python train_meta.py
```

**Note:** `generate_data.py` should be run first before all training.


## Project Structure
- `sg_meta/`: algorithm implementations
  - `data/`: environment settongs and learning hyperparameters.
  - `model.py`: definition of the best response NN model.
  - `agent.py`: implementations of leader and follower classes.
  - `meta.py`: sampling and meta-learning algorithm.
  - `utils.py`: miscellaneous utilities.
- `data/`: data directory for saving generated and learned models.
- `experiments/`: Python scripts for running the experiments.
  - `generate_data.py`: generate the training data.
  - `train_meta.py`: meta learning algorithm implementations.
  - `ave_param.py`: average over parameter space.
  - `ave_output.py`: average over output space.
  - `receding_horizon.py`: receding horizon planning.
  - `zero_guidance.py`: compute follower's trajectory without the leader's guidance.
  - `plot_things.py`: plotting scripts.
- `tests/`: test Python scripts.

**Note:** Meta training and adaptaiton are performed on CPU since we manually implement gradient update for each training iteration. GPU implementation is less efficient.


## Coding specifications
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

To save space, we use `state_dict` to pass a neural network.


### Obstacles
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


### Tunning Parameters
- Meta learning and adaptation hyperparameters are in the `sg_meta/data/parameters.json`.
- For `Param-Ave` and `Output-Ave` training, hyperparameters are defined in the script. They can be different from meta learning hyperparameters.