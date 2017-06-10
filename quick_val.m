net = load(fullfile(vl_rootnn, 'contrib/mcnPyTorch/models/squeezenet1_1-pt-mcn.mat')) ;
dag = dagnn.DagNN.loadobj(net) ;
im_ = single(imresize(imread('peppers.png'), dag.meta.normalization.imageSize(1:2))) ;
dag.conserveMemory = 0 ;

showNames = 0 ;
truncate = 0 ;

if truncate
  names = {dag.layers.name} ;
  names_ = names(end-3:end) ;
  for ii = 1:numel(names_)
    dag.removeLayer(names_{ii}) ;
  end
end
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

