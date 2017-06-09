function [dag, info] = cnn_imagenet_pt_mcn(varargin)
% CNN_IMAGENET_PT_MCN Evaluate imported PyTorch models on ImageNet val set
% (closely based on the mcn cnn_imagenet.m example)

opts.model = 'alexnet-pt-mcn' ;
opts.dataDir = fullfile(vl_rootnn, 'data/datasets/ILSVRC2012') ;
opts.modelDir = fullfile(vl_rootnn, 'contrib/mcnPyTorch/models') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(vl_rootnn, 'data', ['imagenet12-' opts.model]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
opts.train.gpus = [3] ;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
  imdb.imageDir = fullfile(opts.dataDir, 'images');
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end


% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
modelPath = fullfile(opts.modelDir, opts.model) ;
dag = dagnn.DagNN.loadobj(modelPath) ;
dag.addLayer('softmax', dagnn.SoftMax(), dag.layers(end).outputs, 'prediction', {}) ;
dag.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
             {'prediction','label'}, 'top1err') ;
dag.addLayer('top5err', dagnn.Loss('loss', 'topkerror', 'opts', {'topK',5}), ...
             {'prediction','label'}, 'top5err') ;

if strcmp(opts.model, 'alexnet-pt-mcn')
  dag.meta.normalisation.cropSize = 227 / 256 ;
  dag.meta.normalisation.imageSize = [227 227] ;
end


% modify the imdb to skip training images
imdb.images.set(imdb.images.set == 1) = 4 ;
[dag, info] = cnn_train_dag(dag, imdb, getBatchFn(opts, dag.meta), ...
                            'expDir', opts.expDir, 'gpus', opts.train.gpus, ...
                             'numEpochs', 1) ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
bopts = struct('useGpu', numel(opts.train.gpus) > 0, ...
               'imageSize', meta.normalization.imageSize(1:2), ...
               'cropSize', meta.normalization.cropSize) ;
fn = @(x,y) eval_get_batch(bopts,x,y) ;

% -------------------------------------------------------------------------
function varargout = eval_get_batch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
data = getImageBatch(images, opts, 'prefetch', nargout == 0) ;
if nargout > 0
  labels = imdb.images.label(batch) ;
  varargout{1} = {'data', data, 'label', labels} ;
end

% -------------------------------------------------
function data = getImageBatch(imagePaths, varargin)
% -------------------------------------------------
% GETIMAGEBATCH  Load and jitter a batch of images
opts.useGpu = false ;
opts.prefetch = false ;
opts.numThreads = 10 ;

% Options that were used during PyTorch training
% Note: that normalisation must occur after the pixel
% values have been rescaled to [0,1]
opts.cropSize = 224 / 256 ;
opts.imageSize = [224, 224] ;
opts.meanImg = [0.485, 0.456, 0.406] ;
opts.std = [0.229, 0.224, 0.225] ;
opts = vl_argparse(opts, varargin);

args{1} = {imagePaths, ...
           'NumThreads', opts.numThreads, ...
           'Pack', ...
           'Interpolation', 'bilinear', ... % use bilinear to reproduce trainig resize
           'Resize', opts.imageSize(1:2), ...
           'CropSize', opts.cropSize, ...
           'CropAnisotropy', [1 1], ... % preserve aspect ratio
           'CropLocation', 'center'} ; % centre crop for testing

if opts.useGpu, args{end+1} = {'Gpu'} ; end
args = horzcat(args{:}) ;

if opts.prefetch
  vl_imreadjpeg(args{:}, 'prefetch') ;
  data = [] ;
else
  % Normalisation for PyTorch is done a little differently
  data = vl_imreadjpeg(args{:}) ;
  data = data{1} / 255 ; % scale to (almost) [0,1]
  data = bsxfun(@minus, data, permute(opts.meanImg, [1 3 2])) ;
  data = bsxfun(@rdivide, data, permute(opts.std, [1 3 2])) ;
end
