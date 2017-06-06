import ipdb
import torch
import numpy as np
from collections import OrderedDict

# --------------------------------------------------------------------
#                                                             Helpers
# --------------------------------------------------------------------

numerics = (int,float)

mlayerdt = [('name',object),
            ('type',object),
            ('inputs',object),
            ('outputs',object),
            ('params',object),
            ('block',object)]

mparamdt = [('name',object),
            ('value',object)]

minputdt = [('name',object),
            ('size',object)]

def row(x):
    return np.array(x,dtype=float).reshape(1,-1)

def rowcell(x):
    return np.array(x,dtype=object).reshape(1,-1)

def tolist(x):
    """
    Convert x to a Python list. x can be a Torch size tensor, a list, tuple 
    or scalar
    """
    if isinstance(x, torch.Size):
      return [z for z in x]
    elif isinstance(x, (list,tuple)):
      return [z for z in x]
    else:
      return [x]

def pt_tensor_to_array(tensor):
    """
    Convert a PyTorch Tensor to a numpy array.

    (changes the order of dimensions to [width, height, channels, instance])
    """
    dims = tolist(tensor.size())
    raw = tensor.numpy()
    return raw.reshape(dims).transpose()

def dictToMatlabStruct(d): 
    if not d: return np.zeros((0,))
    dt = []
    for x in d.keys():
        pair = (x,object)
        if isinstance(d[x], np.ndarray): pair = (x, type(d[x]))
        dt.append(pair)
    y = np.empty((1,), dtype=dt)
    for x in d.keys():
        y[x][0] = d[x]
    return y

# --------------------------------------------------------------------
#                                              PyTorch Aggregator class
# --------------------------------------------------------------------

class TFTensor(object):
    def __init__(self, name):
        self.name = name
        self.shape = None
        self.value = np.zeros(shape=(0,0), dtype='float32')
        self.bgrInput = False
        self.transposable = True # first two dimensions are spatial

    def transpose(self):
        if self.shape: self.shape = [self.shape[k] for k in [1,0,2,3]]

    def toMatlab(self):
        mparam = np.empty(shape=[1,], dtype=mparamdt)
        mparam['name'][0] = self.name
        mparam['value'][0] = self.value
        return mparam

    def hasValue(self):
        return reduce(mul, self.value.shape, 1) > 0

class PTModel(object):
    def __init__(self):
        self.layers = OrderedDict()
        self.vars = OrderedDict()
        self.params = OrderedDict()

    def add_layer(self, layer):
        ename = layer.name
        while ename in self.layers:
            ipdb.set_trace()
            ename = ename + 'x'
        if layer.name != ename:
            print('Warning: a layer with name {} already found' \
                  ', using {} instead'.format(layer.name, ename))
            layer.name = ename
        for v in layer.inputs:  self.add_var(v)
        for v in layer.outputs: self.add_var(v)
        for p in layer.params: self.add_param(p)
        self.layers[layer.name] = layer

    def add_var(self, name):
        if name not in self.vars:
            self.vars[name] = TFTensor(name)

    def add_param(self, name):
        if name not in self.params:
            self.params[name] = TFTensor(name)

    def renameLayer(self, old, new):
        self.layers[old].name = new
        # reinsert layer with new name -- this mess is to preserve the order
        layers = OrderedDict([(new,v) if k==old else (k,v)
                              for k,v in self.layers.items()])
        self.layers = layers

# --------------------------------------------------------------------
#                                                         Basic Layers
# --------------------------------------------------------------------

class PTLayer(object):
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.params = []
        self.model = None

    def reshape(self, model):
        pass

    def display(self):
        print("Layer \'{}\'".format(self.name))
        print("  +- type: %s" % (self.__class__.__name__))
        print("  +- inputs: %s" % (self.inputs,))
        print("  +- outputs: %s" % (self.outputs,))
        print("  +- params: %s" % (self.params,))

    def transpose(self, model):
        pass

    def setTensor(self, model, all_params):
        print(self.name) # useful sanity check
        ipdb.set_trace()
        assert(False)

    def toMatlab(self):
        mlayer = np.empty(shape=[1,],dtype=mlayerdt)
        mlayer['name'][0] = self.name
        mlayer['type'][0] = None
        mlayer['inputs'][0] = rowcell(self.inputs)
        mlayer['outputs'][0] = rowcell(self.outputs)
        mlayer['params'][0] = rowcell(self.params)
        mlayer['block'][0] = dictToMatlabStruct({})
        return mlayer

class PTElementWise(PTLayer):
    def reshape(self, model):
        for i in range(len(self.inputs)):
            model.vars[self.outputs[i]].shape = \
                model.vars[self.inputs[i]].shape

class PTReLU(PTElementWise):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)

    def setTensor(self, model, all_params):
        pass

    def toMatlab(self):
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.ReLU'
        mlayer['block'][0] = dictToMatlabStruct(
            {'leak': float(0.0) })
        # todo: leak factor
        return mlayer

class PTConv(PTLayer):
    def __init__(self, name, inputs, outputs,
                 num_output,
                 bias_term,
                 pad,
                 kernel_size,
                 stride,
                 dilation,
                 group):

        super().__init__(name, inputs, outputs)

        if len(kernel_size) == 1 : kernel_size = kernel_size * 2
        if len(stride) == 1 : stride = stride * 2
        if len(dilation) == 1 : dilation = dilation * 2
        if len(pad) == 1 : pad = pad * 4
        elif len(pad) == 2 : pad = [pad[0], pad[0], pad[1], pad[1]]

        self.num_output = num_output
        self.bias_term = bias_term
        self.pad = pad
        self.kernel_size = tolist(kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.group = group

        self.params = [name + '_filter']
        if bias_term: self.params.append(name + '_bias')
        self.filter_depth = None

    def display(self):
        super(PTConv, self).display()
        print("  +- filter dimension:", self.filter_depth)
        print("  c- num_output (num filters): %s" % self.num_output)
        print("  c- bias_term: %s" % self.bias_term)
        print("  c- pad: %s" % (self.pad,))
        print("  c- kernel_size: %s" % self.kernel_size)
        print("  c- stride: %s" % (self.stride,))
        print("  c- dilation: %s" % (self.dilation,))
        print("  c- group: %s" % (self.group,))

    def reshape(self, model):
        varin = model.vars[self.inputs[0]]
        varout = model.vars[self.outputs[0]]
        if not varin.shape: return
        varout.shape = getFilterOutputSize(varin.shape[0:2],
                                           self.kernel_size,
                                           self.stride,
                                           self.pad) \
                                           + [self.num_output, varin.shape[3]]
        self.filter_depth = varin.shape[2] / self.group

    def getTransforms(self, model):
        return [[getFilterTransform(self.kernel_size, self.stride, self.pad)]]

    def setTensor(self, model, all_params):
        filters = pt_tensor_to_array(all_params['{}_weight'.format(self.name)])
        while len(filters.shape) < 4 : # handle torch FC tensors
            filters = np.expand_dims(filters,axis=0)
        assert(filters.shape[0] == self.kernel_size[0])
        assert(filters.shape[1] == self.kernel_size[1])
        assert(filters.shape[3] == self.num_output)
        self.filter_depth = filters.shape[2]
        tensors = [filters]
        if self.bias_term:
            bias = pt_tensor_to_array(all_params['{}_bias'.format(self.name)])
            tensors.append(bias)
        for ii, tensor in enumerate(tensors):
            model.params[self.params[ii]].value = tensor
            model.params[self.params[ii]].shape = tensor.shape

    def toMatlab(self):
        size = self.kernel_size + [self.filter_depth, self.num_output]
        mlayer = super(PTConv, self).toMatlab()
        mlayer['type'][0] = u'dagnn.Conv'
        mlayer['block'][0] = dictToMatlabStruct(
            {'hasBias': self.bias_term,
             'size': row(size),
             'pad': row(self.pad),
             'stride': row(self.stride),
             'dilate': row(self.dilation)})
        return mlayer

# --------------------------------------------------------------------
#                                                            BatchNorm
# --------------------------------------------------------------------

class PTBatchNorm(PTLayer):
    def __init__(self, name, inputs, outputs, use_global_stats, 
                                             moving_average_fraction, eps):
        super().__init__(name, inputs, outputs)

        self.use_global_stats = use_global_stats
        self.moving_average_fraction = moving_average_fraction
        self.eps = eps

        self.params = [name + u'_mean',
                       name + u'_variance',
                       name + u'_scale_factor']

    def display(self):
        super().display()
        print("  c- use_global_stats: %s" % (self.use_global_stats,))
        print("  c- moving_average_fraction: %s" % (self.moving_average_fraction,))
        print("  c- eps: %s" % (self.eps))

    def setTensor(self, model, all_params):
        #name_ = name.replace('_'
        gamma = pt_tensor_to_array(all_params['{}_weight'.format(self.name)])
        beta = pt_tensor_to_array(all_params['{}_bias'.format(self.name)])
        mean = pt_tensor_to_array(all_params['{}_running_mean'.format(self.name)])
        var = pt_tensor_to_array(all_params['{}_running_var'.format(self.name)])
        moments = np.vstack((mean, var)).T
        tensors = [gamma, beta, moments]
        for ii, tensor in enumerate(tensors):
            model.params[self.params[ii]].value = tensor
            model.params[self.params[ii]].shape = tensor.shape

    def reshape(self, model):
        shape = model.vars[self.inputs[0]].shape
        mean = model.params[self.params[0]].value
        variance = model.params[self.params[1]].value
        scale_factor = model.params[self.params[2]].value
        for i in range(3): del model.params[self.params[i]]
        self.params = [self.name + u'_mult',
                       self.name + u'_bias',
                       self.name + u'_moments']

        model.addParam(self.params[0])
        model.addParam(self.params[1])
        model.addParam(self.params[2])

        if shape:
            mult = np.ones((shape[2],),dtype='float32')
            bias = np.zeros((shape[2],),dtype='float32')
            model.params[self.params[0]].value = mult
            model.params[self.params[0]].shape = mult.shape
            model.params[self.params[1]].value = bias
            model.params[self.params[1]].shape = bias.shape

        if mean.size:
            moments = np.concatenate(
                (mean.reshape(-1,1) / scale_factor,
                 np.sqrt(variance.reshape(-1,1) / scale_factor + self.eps)),
                axis=1)
            model.params[self.params[2]].value = moments
            model.params[self.params[2]].shape = moments.shape

        model.vars[self.outputs[0]].shape = shape

    def toMatlab(self):
        mlayer = super(PTBatchNorm, self).toMatlab()
        mlayer['type'][0] = u'dagnn.BatchNorm'
        mlayer['block'][0] = dictToMatlabStruct(
            {'epsilon': self.eps})
        return mlayer


class PTPooling(PTLayer):
    def __init__(self, name, inputs, outputs,
                 method,
                 pad,
                 kernel_size,
                 stride):

        super().__init__(name, inputs, outputs)

        if isinstance(kernel_size, numerics): kernel_size = [kernel_size,]
        if isinstance(stride, numerics): stride = [stride,]
        if isinstance(pad, numerics): pad = [pad,]

        if len(kernel_size) == 1 : kernel_size = kernel_size * 2
        if len(stride) == 1 : stride = stride * 2
        if len(pad) == 1 : pad = pad * 4
        elif len(pad) == 2 : pad = [pad[0], pad[0], pad[1], pad[1]]

        self.method = method
        self.pad = tolist(pad)
        self.kernel_size = tolist(kernel_size)
        self.stride = tolist(stride)
        self.pad_corrected = None

    def setTensor(self, model, all_params):
        pass

    def display(self):
        super(PTPooling, self).display()
        print('  +- pad_corrected: %s' % (self.pad_corrected,))
        print('  c- method: ', self.method)
        print('  c- pad: %s' % (self.pad,))
        print('  c- kernel_size: %s' % (self.kernel_size,))
        print('  c- stride: %s' % (self.stride,))

    def reshape(self, model):
        ipdb.set_trace()
        shape = model.vars[self.inputs[0]].shape
        if not shape: return
        # MatConvNet uses a slighly different definition of padding, which we think
        # is the correct one (it corresponds to the filters)
        self.pad_corrected = copy.deepcopy(self.pad)
        for i in [0, 1]:
            self.pad_corrected[1 + i*2] = min(
                self.pad[1 + i*2] + self.stride[i] - 1,
                self.kernel_size[i] - 1)
        model.vars[self.outputs[0]].shape = \
            getFilterOutputSize(shape[0:2],
                                self.kernel_size,
                                self.stride,
                                self.pad_corrected) + shape[2:5]

    def getTransforms(self, model):
        return [[getFilterTransform(self.kernel_size, self.stride, self.pad)]]

    def transpose(self, model):
        self.kernel_size = reorder(self.kernel_size, [1,0])
        self.stride = reorder(self.stride, [1,0])
        self.pad = reorder(self.pad, [2,3,0,1])
        if self.pad_corrected:
            self.pad_corrected = reorder(self.pad_corrected, [2,3,0,1])

    def toMatlab(self):
        mlayer = super().toMatlab()
        if not self.pad_corrected:
            print('Warning: pad correction for layer {} could not be ' \
                 'computed because the layer input shape could not be '  \
                  'determined'.format(self.name))
            self.pad_corrected = self.pad # tmp fix
        mlayer['type'][0] = u'dagnn.Pooling'
        mlayer['block'][0] = dictToMatlabStruct(
            {'method': self.method,
             'poolSize': row(self.kernel_size),
             'stride': row(self.stride),
             'pad': row(self.pad_corrected)})
        return mlayer

# --------------------------------------------------------------------
#                                                               Concat
# --------------------------------------------------------------------

class PTConcat(PTLayer):
    def __init__(self, name, inputs, outputs, concatDim):
        super(PTConcat, self).__init__(name, inputs, outputs)
        self.concatDim = concatDim

    def display(self):
        super().display()
        print("  Concat Dim: ", self.concatDim)

    def setTensor(self, model, all_params):
        pass

    def toMatlab(self):
        mlayer = super(PTConcat, self).toMatlab()
        mlayer['type'][0] = u'dagnn.Concat'
        mlayer['block'][0] = dictToMatlabStruct({'dim': float(self.concatDim)})
        return mlayer

class PTDropout(PTElementWise):
    def __init__(self, name, inputs, outputs, ratio):
        super().__init__(name, inputs, outputs)
        self.ratio = ratio

    def toMatlab(self):
        mlayer = super(PTDropout, self).toMatlab()
        mlayer['type'][0] = u'dagnn.DropOut'
        mlayer['block'][0] = dictToMatlabStruct({'rate': float(self.ratio)})
        return mlayer

    def setTensor(self, model, all_params):
        pass

    def display(self):
        super(PTDropout, self).display()
        print('  c- ratio (dropout rate):', self.ratio)