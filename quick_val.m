%function quick_eval
type = 'double' ;

%modelName = 'resnet50-pt-mcn.mat' ;
%modelName = 'resnext_50_32x4d' ;
modelName = 'resnext_101_64x4d' ;
model = sprintf('%s-pt-mcn.mat', modelName) ;
%modelName = 'resnext_101_32x4d-pt-mcn.mat' ;
%modelName = 'resnext_101_64x4d-pt-mcn.mat' ;

net = load(fullfile(vl_rootnn, 'contrib/mcnPyTorch/models', model)) ;
dag = dagnn.DagNN.loadobj(net) ;
dag.mode = 'test' ; dag.conserveMemory = 0 ;

for ii = 1:numel(dag.params)
  dag.params(ii).value = cast(dag.params(ii).value, type) ;
end

% load image and take centre crop
if 1
  feat_store = load(sprintf('%s-cat.mat', modelName)) ;
  im_ = cast(feat_store.('x0'), type) ;
  im_ = permute(im_, [2 3 1]) ;
else
	imPath = {fullfile(fileparts(mfilename('fullpath')), 'cat.jpg')} ;
	im_ = getImageBatch(imPath) ;
end

type = 'double' ;
for kk = 1:numel(dag.params)
 dag.params(kk).value = cast(dag.params(kk).value, type) ;
end

py_params = load(sprintf('%s-params.mat', modelName)) ;
%disp(mean(im_(:)))
%keyboard

showNames = 0 ; truncate = 0 ;
dag.addLayer('softmax', dagnn.SoftMax(), dag.layers(end).outputs, 'prob', {}) ;
dag.eval({'data', im_}) ;

for ii = 2:numel(dag.layers)-2
  name = dag.layers(ii).name ; inputs = dag.layers(ii).inputs ; outputs = dag.layers(ii).outputs ;
  %fprintf('%d: (%s) layer %s\n', ii, class(dag.layers(ii).block), name) ;
  %for jj = 1:numel(dag.layers(ii).inputs)
    %if showNames, fprintf('+ input: %s\n', dag.layers(ii).inputs{jj}) ; end
  %end
  for jj = 1:numel(dag.layers(ii).outputs)
    mcn = dag.vars(dag.getVarIndex(outputs{jj})).value ;
    if showNames, fprintf('+ output: %s\n', dag.layers(ii).outputs{jj}) ; end

    key = sprintf('x%d', ii) ; py = feat_store.(key) ; py = permute(py, [2 3 1]) ;
    fprintf('%d: (%s): %s vs (%s)\n', ii, class(dag.layers(ii).block), name, key) ;
    rediff = mcn - py ;
    diff = mean(abs(rediff(:))) ;
		%if diff > 1e-2
      %w = sprintf('%s_weight', name) ; b_gain = py_params.(w) ;
      %mb_gain = dag.params(dag.getParamIndex(dag.layers(ii).params{1})).value ;
      %fprintf('weight diff: %g\n', mean(abs(mb_gain(:) - b_gain(:)))) ;

      %w = sprintf('%s_bias', name) ; b_bias = py_params.(w) ;
      %mb_bias = dag.params(dag.getParamIndex(dag.layers(ii).params{2})).value ;
      %fprintf('bias diff: %g\n', mean(abs(mb_bias(:) - b_bias(:)))) ;
			%keyboard
		%end
    fprintf('diff: %g\n', mean(diff)) ;
  end
end

dwarf = copy(dag) ; 
names = {dag.layers.name} ;
numDrop = 240 ;
for ii = 1:numDrop
    dwarf.removeLayer(names{ii}) ;
end

for ii = 1:numel(dwarf.params)
  dwarf.params(ii).value = cast(dwarf.params(ii).value, type) ;
end



dwarf.conserveMemory = 0 ; 
x = cast(permute(feat_store.(sprintf('x%d', numDrop)), [2 3 1]), type) ;
assert(numel(dwarf.getInputs()) == 1, 'should choose a pinchpoint') ;

dwarf.eval({dwarf.getInputs{1}, x}) ;

for ii = 1:numel(dwarf.layers) - 3
  name = dwarf.layers(ii).name ; inputs = dwarf.layers(ii).inputs ; 
  outputs = dwarf.layers(ii).outputs ;
  for jj = 1:numel(dwarf.layers(ii).outputs)
    mcn = dwarf.vars(dwarf.getVarIndex(outputs{jj})).value ;
    if showNames, fprintf('+ output: %s\n', dwarf.layers(ii).outputs{jj}) ; end

    key = sprintf('x%d', ii + numDrop) ; py = feat_store.(key) ; py = permute(py, [2 3 1]) ;
    
    fprintf('%d: (%s): %s vs (%s)\n', ii, class(dwarf.layers(ii).block), name, key) ;
    rediff = mcn - py ;
    diff = mean(abs(rediff(:))) ;
    %diff = mean(abs(mcn(:) - py(:))) ;
    fprintf('diff: %g\n', mean(diff)) ;
    if diff > 1e-4
       keyboard
    end
  end
end

pIdx = dwarf.getParamIndex(dwarf.layers(2).params) ;
mcn_moments = dwarf.params(pIdx(3)).value ;
mcn_moments(:,2) = (mcn_moments(:,2)).^2 ;

run_var = py_params.('features_6_16_0_0_0_1_running_var') ;
run_mean = py_params.('features_6_16_0_0_0_1_running_mean') ;
moments = [ run_mean run_var ] ;


% compute all sigmas
bnorms = find(arrayfun(@(x) isa(dag.layers(x).block, 'dagnn.BatchNorm'), ...
                                             1:numel(dag.layers))) ;
storeMeans = cell(1, numel(bnorms)) ;
storeVars = cell(1, numel(bnorms)) ;

for ii = 1:numel(bnorms)
  bIdx = bnorms(ii) ;
  pIdx = dag.getParamIndex(dag.layers(bIdx).params{3}) ;
  storeMeans{ii} = dag.params(pIdx).value(:,1) ;
  storeVars{ii} = dag.params(pIdx).value(:,2) ;
end

avgMeans = cellfun(@mean, storeMeans) ;
avgVars = cellfun(@mean, storeVars) ;

plot(avgMeans) ;
title('average means') ;
zs_dispFig ;
    
plot(avgVars) ;
title('average means') ;
zs_dispFig ;

%for ii = 1:numel(dag.layers)
  %name = dag.layers(ii).name ;
  %inputs = dag.layers(ii).inputs ;
  %outputs = dag.layers(ii).outputs ;
  %fprintf('-----------\n') ;
  %fprintf('%d: layers %s\n', ii, name) ;
  %for jj = 1:numel(dag.layers(ii).inputs)
    %if showNames, fprintf('+ input: %s\n', dag.layers(ii).inputs{jj}) ; end
  %end
  %for jj = 1:numel(dag.layers(ii).outputs)
    %out = dag.vars(dag.getVarIndex(outputs{jj})).value ;
    %if showNames, fprintf('+ output: %s\n', dag.layers(ii).outputs{jj}) ; end
    %fprintf('+ size: [%d %d %d]\n', size(out)) ;
  %end

   
%end

tmp = importdata('imagenet_class_index.json') ;
data = jsondecode(tmp{1}) ;
labels = cellfun(@(x) {data.(x){2}}, fieldnames(data)) ;

fprintf('output for model %s\n', modelName) ;
scores = dag.vars(dag.getVarIndex('prob')).value ;
scores = squeeze(gather(scores)) ;
[bestScore, best] = max(scores) ;
fprintf('%s (%d), score %.3f\n', labels{best}, best, bestScore) ;
