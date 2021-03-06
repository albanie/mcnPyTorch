function run_pt_benchmarks(varargin)
% do a single pass over the imagenet validation data

  opts.gpus = 2 ;
  opts.batchSize = 32 ;
  opts.useCached = 0 ; % load results from cache if available
  opts.importedModels = {
    'alexnet-pt-mcn', ...
    'vgg11-pt-mcn', ...
    'vgg13-pt-mcn', ...
    'vgg16-pt-mcn', ...
    'vgg19-pt-mcn', ...
    'squeezenet1_0-pt-mcn', ...
    'squeezenet1_1-pt-mcn', ...
    'resnet18-pt-mcn', ...
    'resnet34-pt-mcn', ...
    'resnet101-pt-mcn', ...
    'resnet152-pt-mcn', ...
    'resnext_50_32x4d-pt-mcn', ...
    'resnext_101_32x4d-pt-mcn', ...
    'resnext_101_64x4d-pt-mcn', ...
    'inception_v3-pt-mcn', ...
    'densenet121-pt-mcn', ...
    'densenet161-pt-mcn', ...
    'densenet169-pt-mcn', ...
    'densenet201-pt-mcn', ...
  } ;
  opts = vl_argparse(opts, varargin) ;

  for ii = 1:numel(opts.importedModels)
    model = opts.importedModels{ii} ;
    imagenet_eval(model, opts.batchSize, opts.gpus, opts.useCached) ;
  end

% -------------------------------------------------------
function imagenet_eval(model, batchSize, gpus, useCached)
% -------------------------------------------------------
  [~,info] = cnn_imagenet_pt_mcn('model', model, 'batchSize', ...
                 batchSize, 'gpus', gpus, 'continue', useCached) ;
  top1 = info.val.top1err * 100 ; top5 = info.val.top5err * 100 ;
  fprintf('%s: top-1: %.2f, top-5: %.2f\n', model, top1, top5) ;
