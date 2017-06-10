import sys, os, copy
import argparse
import torch
from pathlib import Path
import numpy as np
import scipy.io
import torchvision
from collections import OrderedDict
import pytorch_utils as pl
from torch.autograd import Variable
from ast import literal_eval as make_tuple
import ipdb

debug = 0
verbose = 1

if debug:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')

    # breakpoint on exception
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
         color_scheme='Linux', call_pdb=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = os.path.expanduser('~/coding/src/zsvision/python')
    sys.path.insert(0, path) # lazy
    from zsvision.zs_iterm import zs_dispFig

if debug:
    sys.argv = ['/Users/samuelalbanie/coding/libs/matconvnets/contrib-matconvnet/contrib/mcnPyTorch/python/import_pytorch.py', '--image-size=[224,224]', '--full-image-size=[256,256]', '--is-torchvision-model=True', 'squeezenet1_0', 'models/squeezenet1_0-pt-mcn.mat']

parser = argparse.ArgumentParser(
           description='Convert model from PyTorch to MatConvNet.')
parser.add_argument('input',
                    type=str,
                    help='The input should be the name of a pytorch model \
                      (if present in pytorch.visionmodels), otherwise it \
                      should be a path to its .pth file')
parser.add_argument('output',
                    type=str,
                    help='Output MATLAB file')
parser.add_argument('--image-size',
                    type=str,
                    nargs='?',
                    default='[224,224]',
                    help='Size of the input image')
parser.add_argument('--full-image-size',
                    type=str,
                    nargs='?',
                    default='[256,256]',
                    help='Size of the full image (from which crops are taken)')
parser.add_argument('--is-torchvision-model',
                    type=bool,
                    nargs='?',
                    default=True,
                    help='is the model part of the torchvision.models')
parser.add_argument('--average-image',
                    type=argparse.FileType('rb'),
                    nargs='?',
                    help='Average image')
parser.add_argument('--average-value',
                    type=str,
                    nargs='?',
                    default=None,
                    help='Average image value')
parser.add_argument('--remove-dropout',
                    dest='remove_dropout',
                    action='store_true',
                    help='Remove dropout layers')
parser.add_argument('--remove-loss',
                    dest='remove_loss',
                    action='store_true',
                    help='Remove loss layers')
parser.add_argument('--append-softmax',
                    dest='append_softmax',
                    action='append',
                    default=[],
                    help='Add a softmax layer after the specified layer')
parser.set_defaults(remove_dropout=False)
parser.set_defaults(remove_loss=False)
args = parser.parse_args()

args.image_size = tuple(make_tuple(args.image_size))
args.full_image_size = tuple(make_tuple(args.full_image_size))

# forward pass to compute pytorch feature sizes
im = scipy.misc.face()
im = scipy.misc.imresize(im, args.image_size)

if debug:
    plt.imshow(im) 
    zs_dispFig()

# model_dir = Path('/users/albanie/coding/libs/convert_torch_to_pytorch/models')
# vgg = model_dir / 'vgg16-397923af.pth'

# params = torch.load(str(vgg))
if args.is_torchvision_model:
    net,flatten_loc = pl.load_valid_pytorch_model(args.input)
else:
    raise ValueError('not yet supported')

transform = pl.ImTransform(args.image_size, (104, 117, 123), (2, 0, 1))
x = Variable(transform(im).unsqueeze(0))
y = net(x)

params = net.state_dict()
feats = pl.compute_intermediate_feats(net, x, flatten_loc)
sizes = [pl.tolist(feat.size()) for feat in feats] 

if verbose:
    for sz in sizes:
        print(sz)

# rename keys to make compatible (duplicates params)
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
        block = construct_layers(children[:6], state)
        layers.extend(block)
        state['in_vars'] = block[-1].outputs

        if downsample:
            state_ = copy.deepcopy(state) ; state_['in_vars'] = id_var
            down_block = construct_layers([children[-1]], state_)
            layers.extend(down_block)
            id_var = down_block[-1].outputs

        cat_name = '{}_cat'.format(name)
        cat_layer = pl.PTConcat(cat_name, [*id_var, *state['in_vars']], [cat_name], 3)
        layers.append(cat_layer)
        state['in_vars'] = cat_layer.outputs

        relu_idx = [child[0] for child in children].index('relu')
        assert relu_idx > 0, 'relu not found'
        relu = construct_layers([children[relu_idx],], state)
        layers.extend(relu) # note that relu is a "one-layer" list

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
            state_ = copy.deepcopy(state) ; state_['in_vars'] = id_var
            down_block,_ = construct_layers([children[-1]], state_)
            layers.extend(down_block)
            id_var = down_block[-1].outputs

        cat_name = '{}_cat'.format(name)
        cat_layer = pl.PTConcat(cat_name, [*id_var, *state['in_vars']], [cat_name], 3)
        state = update_size_info(name, 'mcn-cat', state)
        layers.append(cat_layer)
        state['in_vars'] = cat_layer.outputs

        # add additional ReLU to match model
        name = '{}_id_relu'.format(name)
        state['out_vars'] = [name]
        layers.append(pl.PTReLU(name, state['in_vars'], state['out_vars'])) 
        state = update_size_info(name, 'ReLU', state)

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

        if isinstance(module, torch.nn.modules.conv.Conv2d):
            opts['bias_term'] = bool(module.bias)
            opts['num_output'] = module.out_channels
            opts['kernel_size'] = module.kernel_size
            opts['pad'] = module.padding
            opts['stride'] = module.stride
            opts['dilation'] = module.dilation
            opts['group'] = module.groups
            layers.append(pl.PTConv(*pargs, **opts))
            state = update_size_info(name, 'Conv2d', state)

        elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
            opts['eps'] = module.eps
            opts['use_global_stats'] = module.affine
            opts['moving_average_fraction'] = module.momentum
            layers.append(pl.PTBatchNorm(*pargs, **opts))
            state = update_size_info(name, 'BatchNorm', state)

        elif isinstance(module, torch.nn.modules.activation.ReLU):
            layers.append(pl.PTReLU(*pargs))
            state = update_size_info(name, 'ReLU', state)

        elif isinstance(module, torch.nn.modules.dropout.Dropout):
            if not args.remove_dropout:
                opts['ratio'] = module.p # TODO: check that this shouldn't be 1 - p
                layers.append(pl.PTDropout(*pargs, **opts))
            else:
                state['out_vars'] = state['in_vars']
            state = update_size_info(name, 'Dropout', state)

        elif isinstance(module, torch.nn.modules.pooling.MaxPool2d):
            in_out_sz = state['sizes'][:2]
            state = update_size_info(name, 'MaxPool', state)
            opts['method'] = 'max'
            opts['pad'] = module.padding
            opts['stride'] = module.stride
            opts['kernel_size'] = module.kernel_size
            opts['ceil_mode'] = module.ceil_mode
            opts['sizes'] = in_out_sz
            layers.append(pl.PTPooling(*pargs, **opts))

        elif isinstance(module, torch.nn.modules.pooling.AvgPool2d):
            in_out_sz = state['sizes'][:2]
            state = update_size_info(name, 'AvgPool', state)
            opts['method'] = 'avg'
            opts['pad'] = module.padding
            opts['stride'] = module.stride
            opts['kernel_size'] = module.kernel_size
            opts['ceil_mode'] = module.ceil_mode
            opts['sizes'] = in_out_sz
            layers.append(pl.PTPooling(*pargs, **opts))

        elif isinstance(module, torch.nn.modules.linear.Linear):
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
                              torchvision.models.squeezenet.Fire]:
            prefix_ = state['prefix'] ; state['prefix'] = name
            layers_ = process_custom_module(name, module, state) # soz Guido
            state['prefix'] = prefix_ # restore prefix
            layers.extend(layers_)

        elif isinstance(module, torch.nn.modules.container.Sequential):
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
    return layers, state

mods = list(net.modules())
graph = mods[0].named_children()
state = {'in_vars': ['data'], 'sizes': sizes, 'prefix': ''}
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
    if debug:
        print('---------')
        print('{} has type: {}'.format(name, type(layer)))
        print('name', l.name)
        print('inputs', l.inputs)
        print('outputs', l.outputs)

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

print('Saving network to {}'.format(args.output))
scipy.io.savemat(args.output, mnet, oned_as='column')
