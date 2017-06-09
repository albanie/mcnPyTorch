function feats = get_mcn_features(modelPath, im, imsz)
%GET_MCN_FEATURES - compute all activations for a model

fprintf('running in matlab\n') ;

% hack
path = getenv('LD_LIBRARY_PATH') ;
path = [ '/users/albanie/local/compat/cuda-7.5/lib:/users/albanie/local/compat/cuda-7.5/lib:' path ] ;
setenv('LD_LIBRARY_PATH', path) ;

% setup
addpath('~/coding/libs/matconvnets/contrib-matconvnet/matlab') ;
vl_setupnn ;
addpath(fullfile(vl_rootnn, 'contrib/mcnPyTorch/matlab')) ;
debug = 0 ;

% load model
dag = dagnn.DagNN.loadobj(load(modelPath)) ;

% reshape image
im = single(cell2mat(im)) ; % input im is in pyTorch format (CxHxW)
im = reshape(im, imsz{2}, imsz{3}, []) ;
im = permute(im, [2 1 3]) ;

if debug
  set(0,'DefaultFigureVisible','off') ;
  run '~/coding/libs/vlfeat/toolbox/vl_setup.m'
  run '~/coding/src/zsvision/matlab/zs_setup' ; % visualise inline
  hFig = figure(1) ;
  imagesc(im) ;
  set(hFig, 'Position', [0 0 150 150]) ;
  zs_dispFig ;
  fprintf('mean image value: %.3f\n', mean(im(:))) ;
  fprintf('image size: %d\n', size(im)) ;
  fprintf('image data type: %\ns', class(im)) ;
end

dag.conserveMemory = 0 ;
dag.mode = 'test' ;
dag.eval({'data', im})

for ii = 1:numel(dag.vars)
  feat = dag.vars(ii).value ;
  feats{ii} = feat ;
end

% sanity check
tmp = load('/tmp/labels.mat') ;
labels = tmp.labels ;
dag.addLayer('softmax', dagnn.SoftMax(), dag.layers(end).outputs, 'prob', {}) ;
dag.eval({'data', im})
scores = dag.vars(dag.getVarIndex('prob')).value ;

scores = squeeze(gather(scores)) ;
[bestScore, best] = max(scores) ;
fprintf('%s (%d), score %.3f\n', labels{best}, best, bestScore) ;

