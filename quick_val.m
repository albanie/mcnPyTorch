net = load(fullfile(vl_rootnn, 'contrib/mcnPyTorch/models/resnet34-pt-mcn.mat')) ;
dag = dagnn.DagNN.loadobj(net) ;
im_ = single(imresize(imread('peppers.png'), dag.meta.normalization.imageSize(1:2))) ;
im_ = im_ / 255 ;
im_ = bsxfun(@minus, im_, permute(dag.meta.normalization.averageImage, [3 2 1])) ;
im_ = bsxfun(@rdivide, im_, permute(dag.meta.normalization.imageStd, [3 2 1])) ;
dag.conserveMemory = 0 ;

showNames = 0 ;
truncate = 0;

if truncate
  keep = 29 ;
  names = {dag.layers.name} ;
  drop = numel(names) - keep ;
  names_ = names(end-drop:end) ;
  for ii = 1:numel(names_)
    dag.removeLayer(names_{ii}) ;
  end
end

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
