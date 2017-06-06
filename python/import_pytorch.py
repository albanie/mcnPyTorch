from IPython import get_ipython
ipython = get_ipython()
ipython.magic('load_ext autoreload')
ipython.magic('autoreload 2')

import torch
from pathlib import Path
import numpy as np
import scipy.io
import ipdb
import torchvision
from collections import OrderedDict
import pytorch_layers as pl

# model_dir = Path('/users/albanie/coding/libs/convert_torch_to_pytorch/models')
# vgg = model_dir / 'vgg16-397923af.pth'

# params = torch.load(str(vgg))

# N,C,H,W
# alexnet = models.alexnet(pretrained=True)
# vgg = models.vgg16(pretrained=True)
# params = alexnet.state_dict()

target = 'alexnet'
save_path = '../models/{}-mcn.mat'.format(target)

# --------------------------------------------------------------------
#                                                          Load layers
# --------------------------------------------------------------------
supported_models = [torchvision.models.resnet.ResNet]

if target == 'alexnet':
    net = torchvision.models.alexnet(pretrained=True)
elif target == 'vgg16':
    net = torchvision.models.vgg16(pretrained=True)
elif target == 'resnet50':
    net = torchvision.models.resnet50(pretrained=True)
else:
    raise ValueError('target not recognised')

params = net.state_dict()

# rename keys to make compatible (duplicates params)
tmp = OrderedDict()
for key in params:
    new_name = key.replace('.', '_')
    tmp[new_name] = params[key]
params = tmp 

def process_custom_module(name, module, in_vars):
    layers = []
    if isinstance(module, torchvision.models.resnet.Bottleneck):
        id_var = in_vars
        downsample = hasattr(module, 'downsample') and bool(module.downsample)
        children = list(module.named_children())
        assert len(children) == 7 + downsample, 'unexpected bottleneck size'
        block = construct_layers(children[:6], in_vars=in_vars, prefix=name)
        layers.extend(block)
        in_vars = block[-1].outputs

        if downsample:
            down_block = construct_layers([children[-1]], id_var, prefix=name)
            layers.extend(down_block)
            id_var = down_block[-1].outputs

        cat_name = '{}_cat'.format(name)
        cat_layer = pl.PTConcat(cat_name, [*id_var, *in_vars], [cat_name], 3)
        layers.append(cat_layer)
        in_vars = cat_layer.outputs

        relu_idx = [child[0] for child in children].index('relu')
        assert relu_idx > 0, 'relu not found'
        relu = construct_layers([children[relu_idx],], in_vars=in_vars, prefix=name)
        layers.extend(relu) # note that relu is a "one-layer" list
    else:
        raise ValueError('unrecognised module {}'.format(type(module)))
    return layers

def construct_layers(graph, in_vars, prefix=''):
    layers = [] 
    for name, module in graph:
        name = name.replace('.', '_') # make comatible with MATLAB
        if prefix: name = '{}_{}'.format(prefix, name)
        opts, out_vars = {}, [name,]

        if isinstance(module, torch.nn.modules.conv.Conv2d):
            #print('{} is a Conv2d'.format(name))
            opts['bias_term'] = bool(module.bias)
            opts['num_output'] = module.out_channels
            opts['kernel_size'] = module.kernel_size
            opts['pad'] = module.padding
            opts['stride'] = module.stride
            opts['dilation'] = module.dilation
            opts['group'] = module.groups
            layer = pl.PTConv(name, in_vars, out_vars, **opts)
            layers.append(layer)

        elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
            #print('{} is a BatchNorm'.format(name))
            opts['eps'] = module.eps
            opts['use_global_stats'] = module.affine
            opts['moving_average_fraction'] = module.momentum
            layer = pl.PTBatchNorm(name, in_vars, out_vars, **opts)
            layers.append(layer)

        elif isinstance(module, torch.nn.modules.activation.ReLU):
            #print('{} is a ReLU'.format(name))
            layer = pl.PTReLU(name, in_vars, out_vars)
            layers.append(layer)

        elif isinstance(module, torch.nn.modules.dropout.Dropout):
            #print('{} is a ReLU'.format(name))
            opts['rate'] = module.p # TODO: check that this shouldn't be 1 - p
            layer = pl.PTDropout(name, in_vars, out_vars)
            layers.append(layer)

        elif isinstance(module, torch.nn.modules.pooling.MaxPool2d):
            #print('{} is a max pool'.format(name))
            opts['method'] = 'max'
            opts['pad'] = module.padding
            opts['stride'] = module.stride
            opts['kernel_size'] = module.kernel_size
            layer = pl.PTPooling(name, in_vars, out_vars, **opts)
            layers.append(layer)

        elif isinstance(module, torch.nn.modules.pooling.AvgPool2d):
            #print('{} is an avg pool'.format(name))
            opts['method'] = 'avg'
            opts['pad'] = module.padding
            opts['stride'] = module.stride
            opts['kernel_size'] = module.kernel_size
            layer = pl.PTPooling(name, in_vars, out_vars, **opts)
            layers.append(layer)

        elif isinstance(module, torch.nn.modules.linear.Linear):
            #print('{} is a linear module'.format(name))
            opts['bias_term'] = bool(module.bias)
            opts['num_output'] = module.out_features
            opts['kernel_size'] = [1,1]
            opts['pad'] = [0,0]
            opts['stride'] = [1,1]
            opts['dilation'] = [1,1]
            opts['group'] = 1
            layer = pl.PTConv(name, in_vars, out_vars, **opts)
            layers.append(layer)
            
        elif isinstance(module, torchvision.models.resnet.Bottleneck):
            #print('processing bottleneck: {}'.format(name))
            layers_ = process_custom_module(name, module, in_vars) # soz Guido
            layers.extend(layers_)

        elif isinstance(module, torch.nn.modules.container.Sequential):
            print('flattening sequence: {}'.format(name))
            children = list(module.named_children())
            layers_ = construct_layers(children, in_vars=in_vars, prefix=name)
            layers.extend(layers_)
        elif type(module) in supported_models:
            pass
        else:
            raise ValueError('unrecognised module {}'.format(type(module)))
        in_var = [name,]
    return layers

mods = list(net.modules())
graph = mods[0].named_children()
in_vars = ['data',]
layers = construct_layers(graph, in_vars=in_vars)

# load layers into model and set params
ptmodel = pl.PTModel()
for layer in layers:
    name = layer.name
    ptmodel.add_layer(layer)
    layer.setTensor(ptmodel, params)
    # load parameter values
    print('{} has type: {}'.format(name, type(layer)))

# for l in layers:
   # print(l.name)
   # if not isinstance(l.inputs, list):
       # print(l.name)

# --------------------------------------------------------------------
#                                                        Normalization
# --------------------------------------------------------------------

# standard normalization for vision models in pytorch
average_image = np.array((0.485, 0.456, 0.406),dtype='float')
image_std = np.array((0.229, 0.224, 0.225), dtype='float') 

minputs = np.empty(shape=[0,], dtype=pl.minputdt)
dataShape = [224,224,3] # hardcode as a temp fix
fullImageSize = [256, 256]
print('Input image data tensor shape:', dataShape)
print('Full input image size:', fullImageSize)

mnormalization = {
  'imageSize': pl.row(dataShape),
  'averageImage': average_image,
  'imageStd': image_std, 
  'interpolation': 'bilinear',
  'keepAspect': True,
  'border': pl.row([0,0]),
  'cropSize': 1.0}

# --------------------------------------------------------------------
#                                                    Convert to MATLAB
# --------------------------------------------------------------------

# net.meta
meta_dict = {'inputs': minputs.reshape(1,-1), 'normalization': mnormalization}
mmeta = pl.dictToMatlabStruct(meta_dict)

# This object should stay a dictionary and not a NumPy array due to
# how NumPy saves to MATLAB
mnet = {'layers': np.empty(shape=[0,], dtype=pl.mlayerdt),
        'params': np.empty(shape=[0,], dtype=pl.mparamdt),
        'meta': mmeta}

for layer in ptmodel.layers.values():
    mnet['layers'] = np.append(mnet['layers'], layer.toMatlab(), axis=0)
 
for param in ptmodel.params.values():
    mnet['params'] = np.append(mnet['params'], param.toMatlab(), axis=0)

# to row
mnet['layers'] = mnet['layers'].reshape(1,-1)
mnet['params'] = mnet['params'].reshape(1,-1)

# --------------------------------------------------------------------
#                                                          Save output
# --------------------------------------------------------------------

print('Saving network to {}'.format(save_path))
scipy.io.savemat(save_path, mnet, oned_as='column')
