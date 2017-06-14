function quick_val2

modelName = 'resnet50' ;
%modelName = 'resnext_50_32x4d' ;
%modelName = 'resnext_101_64x4d' ;
model = sprintf('%s-pt-mcn.mat', modelName) ;

net = load(fullfile(vl_rootnn, 'contrib/mcnPyTorch/models', model)) ;
dag = dagnn.DagNN.loadobj(net) ;
imPath = {fullfile(fileparts(mfilename('fullpath')), 'cat.jpg')} ;
im_ = getImageBatch(imPath) ;

%im_ = single(imresize(imread('peppers.png'), dag.meta.normalization.imageSize(1:2))) ;
%im_ = im_ / 255 ;
%im_ = bsxfun(@minus, im_, permute(dag.meta.normalization.averageImage, [3 2 1])) ;
%im_ = bsxfun(@rdivide, im_, permute(dag.meta.normalization.imageStd, [3 2 1])) ;
dag.conserveMemory = 0 ;
showNames = 0 ;

dag.addLayer('softmax', dagnn.SoftMax(), dag.layers(end).outputs, 'prob', {}) ;
dag.eval({'data', im_}) ;


for ii = 1:numel(dag.layers)
  name = dag.layers(ii).name ;
  inputs = dag.layers(ii).inputs ;
  outputs = dag.layers(ii).outputs ;
  fprintf('-----------\n') ;
  fprintf('%d: layers %s\n', ii, name) ;
  for jj = 1:numel(dag.layers(ii).inputs)
    if showNames, fprintf('+ input: %s\n', dag.layers(ii).inputs{jj}) ; end
  end
  for jj = 1:numel(dag.layers(ii).outputs)
    out = dag.vars(dag.getVarIndex(outputs{jj})).value ;
    if showNames, fprintf('+ output: %s\n', dag.layers(ii).outputs{jj}) ; end
    fprintf('+ size: [%d %d %d]\n', size(out)) ;
  end
end

tmp = importdata('imagenet_class_index.json') ;
data = jsondecode(tmp{1}) ;
labels = cellfun(@(x) {data.(x){2}}, fieldnames(data)) ;


scores = dag.vars(dag.getVarIndex('prob')).value ;
scores = squeeze(gather(scores)) ;
[bestScore, best] = max(scores) ;
fprintf('%s (%d), score %.3f\n', labels{best}, best, bestScore) ;

% -------------------------------------------------
function data = getImageBatch(imagePaths, varargin)
% -------------------------------------------------
% Options that were used during PyTorch training
% Note: that normalisation must occur after the pixel
% values have been rescaled to [0,1]
opts.cropSize = 224 / 256 ;
opts.imageSize = [224, 224] ;
opts.meanImg = [0.485, 0.456, 0.406] ;
opts.std = [0.229, 0.224, 0.225] ;
opts = vl_argparse(opts, varargin);

args = {imagePaths, ...
        'Pack', ...
        'Interpolation', 'bilinear', ... % use bilinear to reproduce trainig resize
        'Resize', opts.imageSize(1:2), ...
        'CropSize', opts.cropSize, ...
        'CropAnisotropy', [1 1], ... % preserve aspect ratio
        'CropLocation', 'center'} ; % centre crop for testing

data = vl_imreadjpeg(args{:}) ;
data = data{1} / 255 ; % scale to (almost) [0,1]
data = bsxfun(@minus, data, permute(opts.meanImg, [1 3 2])) ;
data = bsxfun(@rdivide, data, permute(opts.std, [1 3 2])) ;
