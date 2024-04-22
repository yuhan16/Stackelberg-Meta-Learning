import os
import json
import torch
from . import config_data_dir


class BRNet(torch.nn.Module):   # 2 hidden layers
    """
    BRNet class defines and trains a NN for BR model.
    """
    def __init__(self) -> None:
        super(BRNet, self).__init__()

        param = json.load( open(os.path.join(config_data_dir, 'parameters.json')) )
        self.dimx, self.dimua, self.dimub = param['dimx'], param['dimua'], param['dimub']
        
        self.linear1 = torch.nn.Linear(self.dimx+self.dimua, 25)
        self.linear2 = torch.nn.Linear(25, 25)
        self.linear3 = torch.nn.Linear(25, self.dimub)
        self.activation = torch.nn.ReLU()   # or tanh or sigmoid
        
        # random initialization 
        self.linear1.weight.data.normal_(mean=0., std=.1)
        self.linear1.bias.data.normal_(0., .1)
        self.linear2.weight.data.normal_(0, .1)
        self.linear2.bias.data.normal_(0, .1)
        self.linear3.weight.data.normal_(0, .1)
        self.linear3.bias.data.normal_(0, .1)
        
        # constant initialization for testing
        # self.linear1.weight.data.fill_(.1)     
        # self.linear1.bias.data.fill_(.1)
        # self.linear2.weight.data.fill_(.1)
        # self.linear2.bias.data.fill_(.1)
        # self.linear3.weight.data.fill_(.1)
        # self.linear3.bias.data.fill_(.1)
    

    def forward(self, x, a):
        if x.ndim > 1:
            y = torch.cat((x, a), dim=1)
        else:
            y = torch.cat((x, a), dim=0)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.linear2(y)
        y = self.activation(y)
        y = self.linear3(y)
        return y
    

    def compute_input_jac(self, x, a):
        """
        This function computes the jacobian of brnet w.r.t. input x and a.
        """
        # register hook for inner layer outpuut
        y = []  # y[i] is a 2d array
        def forward_hook(model, input, output):
            y.append( output.detach() )
        h1 = self.linear1.register_forward_hook(forward_hook)
        h2 = self.linear2.register_forward_hook(forward_hook)
        h3 = self.linear3.register_forward_hook(forward_hook)
        _ = self.forward(x, a)
        h1.remove()
        h2.remove()
        h3.remove()
        
        def d_activation(y):
            """
            This function computes derivative of activation functions. can be relu, tanh, sigmoid.
            Input is a 1d array, output is nxn matrix.
            """
            #df = torch.diag(1 - torch.tanh(y)**2)  # for tanh(x)
            df = torch.diag(1. * (y > 0))           # for relu(x)
            return df
        def d_normalize(y):
            """
            This function computes the derivative of normalization functions. can be sigmoid, tanh.
            """
            df = torch.diag(y*(1-y))    # for sigmoid, need dot product
            #df = torch.diag(1-y**2)     # for tanh
            return df
        p = self.state_dict()
        jac_x = p['linear3.weight'] @ d_activation(y[1]) @ p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, : self.dimx]
        jac_a = p['linear3.weight'] @ d_activation(y[1]) @ p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, self.dimx: ]
        return jac_x, jac_a
    

    def get_grad_dict(self):
        dp_dict = {}
        for n, p in self.named_parameters():
            dp_dict[n] = p.grad.detach().cpu()
        return dp_dict


    def get_zero_grad_dict(self):
        dp_dict = {} 
        for n, p in self.named_parameters():
            dp_dict[n] = torch.zeros_like(p.data)
        return dp_dict


    def get_intermediate_output(self, x, a):
        """
        This function gets the output of every Linear layer.
        """
        y = []
        def forward_hook(model, input, output):
            y.append( output.detach() )
        h1 = self.linear1.register_forward_hook(forward_hook)
        h2 = self.linear2.register_forward_hook(forward_hook)
        h3 = self.linear3.register_forward_hook(forward_hook)
        _ = self.forward(x, a)
        h1.remove()
        h2.remove()
        h3.remove()
        return y 
    


class BRNet1(torch.nn.Module):  # 1 hidden layer
    """
    BRNet class defines and trains a NN for BR model.
    """
    def __init__(self) -> None:
        super(BRNet, self).__init__()

        param = json.load( open(os.path.join(config_data_dir, 'parameters.json')) )
        self.dimx, self.dimua, self.dimub = param['dimx'], param['dimua'], param['dimub']
    
        self.linear1 = torch.nn.Linear(self.dimx+self.dimua, 15)
        self.linear2 = torch.nn.Linear(15, self.dimub)
        self.activation = torch.nn.ReLU()   # or tanh or sigmoid
        
        # random initialization 
        self.linear1.weight.data.normal_(mean=0., std=.1)
        self.linear1.bias.data.normal_(0., .1)
        self.linear2.weight.data.normal_(0, .1)
        self.linear2.bias.data.normal_(0, .1)
        
        # constant initialization for testing
        # self.linear1.weight.data.fill_(.1)     
        # self.linear1.bias.data.fill_(.1)
        # self.linear2.weight.data.fill_(.1)
        # self.linear2.bias.data.fill_(.1)
    

    def forward(self, x, a):
        if x.ndim > 1:
            y = torch.cat((x, a), dim=1)
        else:
            y = torch.cat((x, a), dim=0)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.linear2(y)
        return y
    

    def compute_input_jac(self, x, a):
        """
        This function computes the jacobian of brnet w.r.t. input x and a.
        """
        # register hook for inner layer outpuut
        y = []  # y[i] is a 2d array
        def forward_hook(model, input, output):
            y.append( output.detach() )
        h1 = self.linear1.register_forward_hook(forward_hook)
        _ = self.forward(x, a)
        h1.remove()
        
        def d_activation(y):
            """
            This function computes derivative of activation functions. can be relu, tanh, sigmoid.
            Input is a 1d array, output is n x n matrix.
            """
            #df = torch.diag(1 - torch.tanh(y)**2)  # for tanh(x)
            df = torch.diag(1. * (y > 0))           # for relu(x)
            return df
        p = self.state_dict()
        jac_x = p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, : self.dimx]
        jac_a = p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, self.dimx: ]
        return jac_x, jac_a
    

    def get_grad_dict(self):
        dp_dict = {}
        for n, p in self.named_parameters():
            dp_dict[n] = p.grad.detach().cpu()
        return dp_dict


    def get_zero_grad_dict(self):
        dp_dict = {} 
        for n, p in self.named_parameters():
            dp_dict[n] = torch.zeros_like(p.data)
        return dp_dict


    def get_intermediate_output(self, x, a):
        """
        This function gets the output of every Linear layer.
        """
        y = []
        def forward_hook(model, input, output):
            y.append( output.detach() )
        h1 = self.linear1.register_forward_hook(forward_hook)
        _ = self.forward(x, a)
        h1.remove()
        return y 

