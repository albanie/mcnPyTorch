import torch
from torch import nn
import scipy.io
from torch import autograd
from torch.autograd import Variable

eps = 1e-5

model = 'resnext_101_64x4d'
d = scipy.io.loadmat('../{}-cat.mat'.format(model))
d2 = scipy.io.loadmat('../{}-params.mat'.format(model))
#input = autograd.Variable(torch.randn(20, 100, 35, 45))
x_ = d['x245']
x = Variable(torch.Tensor(x_))
x = torch.unsqueeze(x,0)

mult = d2['features_6_16_0_0_0_4_weight'] ;
bias = d2['features_6_16_0_0_0_4_bias'] ;
mean = d2['features_6_16_0_0_0_4_running_mean'] ;
var = d2['features_6_16_0_0_0_4_running_var'] ;


# height = 20
# width = 20
channels = 1024
sigma = 1.1
var_scale = sigma**2 ;
#x = autograd.Variable(torch.ones([1,channels,height,width]) * 0.5)
m = nn.BatchNorm2d(channels, affine=False, eps=eps)

def prep(x):
    x = torch.Tensor(x)
    return torch.squeeze(x)

m.weight = nn.Parameter(prep(mult.T))
m.bias = nn.Parameter(prep(bias.T))
m.running_mean = prep(mean.T)
m.running_var = prep(var.T)

frozen = m.eval()
y = m(x)

res = y.data.numpy()

print('sum: {:.9g}'.format(res.sum()))
