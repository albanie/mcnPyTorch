function feats = get_mcn_features(im, imsz)

fprintf('running in matlab\n') ;

% setup
addpath('~/coding/libs/matconvnets/contrib-matconvnet/matlab') ;
vl_setupnn ;
addpath(fullfile(vl_rootnn, 'contrib/mcnPyTorch/matlab')) ;
debug = 0 ;

% load model
model = 'alexnet-mcn.mat' ;
modelDir = fullfile(vl_rootnn, 'contrib/mcnPyTorch/models') ;
net = load(fullfile(modelDir, model)) ;
dag = dagnn.DagNN.loadobj(net) ;
dag.addLayer('softmax', dagnn.SoftMax(), dag.layers(end).outputs, 'prob', {}) ;

% reshape image
im = single(cell2mat(im)) ;
im = reshape(im, imsz{1}, imsz{2}, []) ;
im = permute(im, [2 1 3]) ;

if debug
  set(0,'DefaultFigureVisible','off') ;
  run '~/coding/src/zsvision/matlab/zs_setup' ; % visualise inline
  disp(size(im))
  disp(class(im))
  imagesc(im) ;
  zs_dispFig ;
  disp(mean(im(:)))
end

dag.conserveMemory = 0 ;
dag.eval({'data', im})

for ii = 1:numel(dag.vars)
  feat = dag.vars(ii).value ;
  feats{ii} = feat ;
end
