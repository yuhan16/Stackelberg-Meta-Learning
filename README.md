# sg-meta
This repo is for sg meta learning project


## Code specifications
- One global parameter file, which defines type-related information.
- Four classes to implement different functions: `Leader`, `Follower`, `BRNet`, `Meta`.
  - `Leader`: compute leader's objective; solve leader's decision problem.
  - `Follower`: compute follower's objective; compute follower's best response.
  - `BRNet`: defines the best response NN.
  - `Meta`: sample and algorithm.

Leader's obj parameter `alp`. Follower's obj parameter `beta`.

We use numpy array to store trajectories. For example, x_traj[t, :] represents x_t.
We use different variables to represent trajectories. For example, x_traj and a_traj
- `x_traj` has dimension T+1, `x_0, ..., x_T`. `a_traj` and `b_traj` have dimension T, `a_0, ..., a_{T-1}`.

We use numpy array to store samples. For example, x_samp[i, :] represents i-th sample in x.
We use different variables to represent samples. For example, x_samp and a_samp.
We need to convert to tensor for pytorch computation if necessary.

Use 1d array for computation when there is no trajectory requirement.

In meta-learning, `brnet` only stores parameter value. `.grad` is useless.
The NN in `br_list` stores the same parameter as `brnet` and updated gradient.

## function structure
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


## Notice
- When writing constraints, do not use for loop. Write all constraints as a whole. This is because the computer needs a function to compute constraints in each iteration. If using for loop, we need to write N different function definitions explicitly.