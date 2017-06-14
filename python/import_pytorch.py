import os
import sys
import copy
import ipdb
import torch
import argparse
import scipy.io
import torchvision
import numpy as np
from torch import nn
from pathlib import Path
import pytorch_utils as pl
from collections import OrderedDict
from torch.autograd import Variable
from ast import literal_eval as make_tuple

# parse args
parser = pl.set_conversion_kwargs()
args = parser.parse_args(sys.argv[1:])

# load model
if args.model_def and args.model_weights:
    model_paths = {'def': args.model_def, 'weights': args.model_weights}
net,flatten_loc = pl.load_pytorch_model(args.pytorch_model, paths=model_paths)
params = net.state_dict()

# forward pass to compute pytorch feature sizes
args.image_size = tuple(make_tuple(args.image_size))
args.full_image_size = tuple(make_tuple(args.full_image_size))
im = scipy.misc.imresize(scipy.misc.face(), args.image_size)
transform = pl.ImTransform(args.image_size, (104, 117, 123), (2, 0, 1))
x = Variable(transform(im).unsqueeze(0))
feats = pl.compute_intermediate_feats(net.eval(), x, flatten_loc)
sizes = [pl.tolist(feat.size()) for feat in feats] 

for sz in sizes:
    print(sz)

# rename keys to make compatible with MATLAB
tmp = OrderedDict()
for key in params:
    new_name = key.replace('.', '_')
    tmp[new_name] = params[key]
params = tmp 

def process_custom_module(name, module, state):
    layers = []
    if isinstance(module, torchvision.models.resnet.Bottleneck):
        id_var = state['in_vars']
        downsample = hasattr(module, 'downsample') and bool(module.downsample)
        children = list(module.named_children())
        assert len(children) == 7 + downsample, 'unexpected bottleneck size'
        state['prefix'] = name

        # insert repeated ReLU
        relu_ = children[6]
        is_relu = isinstance(relu_[1], nn.modules.activation.ReLU)
        assert is_relu, 'unusual relu location'
        children.insert(4, ('relu2', relu_[1]))
        children.insert(2, ('relu1', relu_[1]))
        block, state = construct_layers(children[:8], state)
        layers.extend(block)
        state['in_vars'] = block[-1].outputs

        if downsample:
            prev = state['in_vars'] ; state['in_vars'] = id_var 
            down_block,_ = construct_layers([children[-1]], state)
            layers.extend(down_block)
            state['in_vars'] = prev
            id_var = down_block[-1].outputs


        merge_name = '{}_merge'.format(name)
        merge_layer = pl.PTSum(merge_name, [*id_var, *state['in_vars']], [merge_name])
        state = update_size_info(name, 'mcn-sum', state)
        layers.append(merge_layer)
        state['in_vars'] = merge_layer.outputs

        # add additional ReLU to match model
        name = '{}_id_relu'.format(name)
        state['out_vars'] = [name]
        layers.append(pl.PTReLU(name, state['in_vars'], state['out_vars'])) 
        state = update_size_info(name, 'ReLU', state)
        state['in_vars'] = state['out_vars']

    elif isinstance(module, torchvision.models.resnet.BasicBlock):
        id_var = state['in_vars']
        downsample = hasattr(module, 'downsample') and bool(module.downsample)
        children = list(module.named_children())
        assert len(children) == 5 + downsample, 'unexpected bottleneck size'
        state['prefix'] = name

        block, state = construct_layers(children[:5], state)
        layers.extend(block)
        state['in_vars'] = block[-1].outputs

        if downsample:
            prev = state['in_vars'] ; state['in_vars'] = id_var 
            down_block,_ = construct_layers([children[-1]], state)
            layers.extend(down_block)
            state['in_vars'] = prev
            id_var = down_block[-1].outputs

        merge_name = '{}_merge'.format(name)
        merge_layer = pl.PTSum(merge_name, [*id_var, *state['in_vars']], [merge_name])
        state = update_size_info(name, 'mcn-sum', state)
        layers.append(merge_layer)
        state['in_vars'] = merge_layer.outputs

        # add additional ReLU to match model
        name = '{}_id_relu'.format(name)
        state['out_vars'] = [name]
        layers.append(pl.PTReLU(name, state['in_vars'], state['out_vars'])) 
        state = update_size_info(name, 'ReLU', state)
        state['in_vars'] = state['out_vars']

    elif isinstance(module, torchvision.models.squeezenet.Fire):
        state['prefix'] = name # process squeeze block first
        children = list(module.named_children())
        assert len(children) == 6 , 'unexpected fire size'
        squeeze_block,_ = construct_layers(children[:2], state)
        layers.extend(squeeze_block)
        expand1x1,_ = construct_layers(children[2:4], state) # expand 1x1
        layers.extend(expand1x1)
        out1 = expand1x1[-1].outputs
        state['in_vars'] = squeeze_block[-1].outputs
        expand3x3,_ = construct_layers(children[4:], state) # expand 3x3
        layers.extend(expand3x3)
        out3 = expand3x3[-1].outputs
        cat_name = '{}_cat'.format(name)
        cat_layer = pl.PTConcat(cat_name, [*out1, *out3], [cat_name], 3)
        state = update_size_info(name, 'mcn-cat', state)
        layers.append(cat_layer)
        state['out_vars'] = cat_layer.outputs
    elif pl.is_lambda_map(list(module.children())[0]):
        id_var = state['in_vars']
        children = list(module.named_children())
        assert pl.is_lambda_reduce(children[1][1]), 'invalid map reduce pair'
        _map, _reduce = children[0][1].lambda_func, children[1][1].lambda_func
        if _map(1) != 1: raise ValueError('only identity map supported')
        if _reduce(1,1) != 2: raise ValueError('only summation reduce supported')
        state['prefix'] = '{}_{}'.format(name, children[0][0]) # handle skipped prefix
        trunk = list(children[0][1].named_children())[0]
        block, state = construct_layers([trunk], state)
        layers.extend(block)
        state['in_vars'] = block[-1].outputs

        projection = list(children[0][1].named_children())[1]
        if not hasattr(projection[1], 'lambda_func'): # projection exists
            prev = state['in_vars'] ; state['in_vars'] = id_var 
            down_block,_ = construct_layers([projection], state)
            layers.extend(down_block)
            state['in_vars'] = prev
            id_var = down_block[-1].outputs
        else:
            msg = 'non projection should be identity'
            assert projection[1].lambda_func(1) == 1, msg

        merge_name = '{}_merge'.format(name)
        merge_layer = pl.PTSum(merge_name, [*id_var, *state['in_vars']], [merge_name])
        state = update_size_info(name, 'mcn-sum', state)
        layers.append(merge_layer)
        state['in_vars'] = merge_layer.outputs

        # add additional ReLU to match model
        msg = 'expected ReLU at end of lambda unit'
        assert isinstance(children[-1][1], nn.modules.activation.ReLU), msg

        name = '{}_id_relu'.format(name)
        state['out_vars'] = [name]
        layers.append(pl.PTReLU(name, state['in_vars'], state['out_vars'])) 
        state = update_size_info(name, 'ReLU', state)
        state['in_vars'] = state['out_vars']
    else:
        raise ValueError('unrecognised module {}'.format(type(module)))
    return layers

def update_size_info(name, module, state, pop_first=1):
    """
    print size summary, perform some sanity checks and (by default) pops
    the leading size
    """
    print('{}: {}'.format(name, module))
    in_sz, out_sz = state['sizes'][:2]
    print(' +  size: {} -> {}'.format(in_sz, out_sz))
    if module in ['ReLU', 'BatchNorm']:
        if in_sz != out_sz:
            ipdb.set_trace()
        assert in_sz == out_sz, 'sizes should match for {}'.format(module)
    if pop_first:
        state['sizes'].pop(0)
    return state

def flatten_layers(name, layers, state) :
    """
    Flattening is done in the network class, rather 
    than in the moudles with pytorch, so we need to 'catch' this event 
    and reproduce it in the mcn sense.  The reshape is essentially free, 
    but the permute can add some overhead
    """
    if state['prefix']: name = '{}_{}'.format(state['prefix'], name)
    name_perm = '{}_permute'.format(name)
    state['out_vars'] = [name_perm]
    pargs = [name_perm, state['in_vars'], state['out_vars']]
    layers.append(pl.PTPermute(*pargs, order=[2,1,3,4]))
    state['in_vars'] = state['out_vars']

    name_flat = '{}_flatten'.format(name)
    state['out_vars'] = [name_flat]
    pargs = [name_flat, state['in_vars'], state['out_vars']]
    layers.append(pl.PTFlatten(*pargs, axis=3))
    state['in_vars'] = state['out_vars']
    #state = update_size_info(name, 'mcn-flatten', state)
    return layers, state

def construct_layers(graph, state):
    """
    `state`: a dictionary which is carried through the graph construction
    `opts`: a dict of opts for the current layer
    """
    layers = [] 
    for name, module in graph:

        name = name.replace('.', '_') # make comatible with MATLAB

        if name == 'classifier' and flatten_loc == 'classifier': 
            layers, state = flatten_layers(name, layers, state)

        opts = {} 
        if state['prefix']: name = '{}_{}'.format(state['prefix'], name)
        state['out_vars'] = [name]
        pargs = [name, state['in_vars'], state['out_vars']]

        if isinstance(module, nn.modules.conv.Conv2d):
            opts['bias_term'] = bool(module.bias)
            opts['num_output'] = module.out_channels
            opts['kernel_size'] = module.kernel_size
            opts['pad'] = module.padding
            opts['stride'] = module.stride
            opts['dilation'] = module.dilation
            opts['group'] = module.groups
            layers.append(pl.PTConv(*pargs, **opts))
            state = update_size_info(name, 'Conv2d', state)

        elif isinstance(module, nn.modules.batchnorm.BatchNorm2d):
            opts['eps'] = module.eps
            opts['use_global_stats'] = module.affine
            opts['momentum'] = module.momentum
            layers.append(pl.PTBatchNorm(*pargs, **opts))
            state = update_size_info(name, 'BatchNorm', state)

        elif isinstance(module, nn.modules.activation.ReLU):
            layers.append(pl.PTReLU(*pargs))
            state = update_size_info(name, 'ReLU', state)

        elif isinstance(module, nn.modules.dropout.Dropout):
            if not args.remove_dropout:
                opts['ratio'] = module.p # TODO: check that this shouldn't be 1 - p
                layers.append(pl.PTDropout(*pargs, **opts))
            else:
                state['out_vars'] = state['in_vars']
            state = update_size_info(name, 'Dropout', state)

        elif isinstance(module, nn.modules.pooling.MaxPool2d):
            in_out_sz = state['sizes'][:2]
            state = update_size_info(name, 'MaxPool', state)
            opts['method'] = 'max'
            opts['pad'] = module.padding
            opts['stride'] = module.stride
            opts['kernel_size'] = module.kernel_size
            opts['ceil_mode'] = module.ceil_mode
            opts['sizes'] = in_out_sz
            layers.append(pl.PTPooling(*pargs, **opts))

        elif isinstance(module, nn.modules.pooling.AvgPool2d):
            in_out_sz = state['sizes'][:2]
            state = update_size_info(name, 'AvgPool', state)
            opts['method'] = 'avg'
            opts['pad'] = module.padding
            opts['stride'] = module.stride
            opts['kernel_size'] = module.kernel_size
            opts['ceil_mode'] = module.ceil_mode
            opts['sizes'] = in_out_sz
            layers.append(pl.PTPooling(*pargs, **opts))

        elif isinstance(module, nn.modules.linear.Linear):
            state = update_size_info(name, 'Linear', state)
            opts['bias_term'] = bool(module.bias)
            opts['filter_depth'] = module.in_features
            opts['num_output'] = module.out_features
            opts['kernel_size'] = [1,1]
            opts['pad'] = [0,0]
            opts['stride'] = [1,1]
            opts['dilation'] = [1,1]
            opts['group'] = 1
            layers.append(pl.PTConv(*pargs, **opts))
            

        elif type(module) in [torchvision.models.resnet.Bottleneck, 
                              torchvision.models.resnet.BasicBlock, 
                              torchvision.models.squeezenet.Fire] or \
                              pl.has_lambda_child(module):
            prefix_ = state['prefix'] ; state['prefix'] = name
            layers_ = process_custom_module(name, module, state) # soz Guido
            state['prefix'] = prefix_ # restore prefix
            layers.extend(layers_)

        elif isinstance(module, nn.modules.container.Sequential):
            print('flattening sequence: {}'.format(name))
            children = list(module.named_children())
            prefix_ = state['prefix'] ; state['prefix'] = name
            layers_,state = construct_layers(children, state)
            state['out_vars'] = layers_[-1].outputs 
            state['prefix'] = prefix_ # restore prefix
            layers.extend(layers_)

        elif type(module) in [torchvision.models.resnet.ResNet,]:
            pass
        else:
            raise ValueError('unrecognised module {}'.format(type(module)))
        state['in_vars'] = state['out_vars']

        if name == '8':
            ipdb.set_trace()
    return layers, state

mods = list(net.modules())
graph = mods[0].named_children()
state = {'in_vars': ['data'], 'sizes': sizes[:], 'prefix': ''}
layers, state = construct_layers(graph, state)

if flatten_loc == 'end':
    name = 'final'
    layers, state = flatten_layers(name, layers, state)

# load layers into model and set params
ptmodel = pl.PTModel()
for layer in layers:
    name = layer.name
    ptmodel.add_layer(layer)
    # load parameter values
    layer.setTensor(ptmodel, params)
    print('---------')
    print('{} has type: {}'.format(name, type(layer)))
    print('name', layer.name)
    print('inputs', layer.inputs)
    print('outputs', layer.outputs)

# --------------------------------------------------------------------
#                                                        Normalization
# --------------------------------------------------------------------

# standard normalization for vision models in pytorch
average_image = np.array((0.485, 0.456, 0.406),dtype='float')
image_std = np.array((0.229, 0.224, 0.225), dtype='float') 

minputs = np.empty(shape=[0,], dtype=pl.minputdt)
dataShape = args.image_size
print('Input image data tensor shape:', dataShape)
print('Full input image size:', args.full_image_size)

mnormalization = {
  'imageSize': pl.row(dataShape),
  'averageImage': average_image,
  'imageStd': image_std, 
  'interpolation': 'bilinear',
  'keepAspect': True,
  'border': pl.row([0,0]),
  'cropSize': 1.0}

fw = max(args.full_image_size[0], dataShape[1])
fh = max(args.full_image_size[0], dataShape[0])
mnormalization['border'] = max([float(fw - dataShape[1]),
                  float(fh - dataShape[0])])
mnormalization['cropSize'] = min([float(dataShape[1]) / fw,
                float(dataShape[0]) / fh])

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
#                                                  Save converted model
# --------------------------------------------------------------------

print('Saving network to {}'.format(args.mcn_model))
scipy.io.savemat(args.mcn_model, mnet, oned_as='column')
