import os 


# generate data directory in the project root directory
project_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_root, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# generate training data directory
train_data_dir = os.path.join(data_dir, 'train')
if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)

# generate plot data directory
plot_data_dir = os.path.join(data_dir, 'plots')
if not os.path.exists(plot_data_dir):
    os.makedirs(plot_data_dir)

# genetate tmp/test data directory
tmp_data_dir = os.path.join(data_dir, 'tmp')
if not os.path.exists(tmp_data_dir):
    os.makedirs(tmp_data_dir)

# get configuration data directory
config_data_dir = os.path.join(os.path.dirname(__file__), 'data')