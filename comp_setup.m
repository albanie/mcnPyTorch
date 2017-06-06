model = 'alexnet-mcn.mat' ;
modelDir = fullfile(vl_rootnn, 'contrib/mcnPyTorch/models') ;
net = load(fullfile(modelDir, model)) ;

dag = dagnn.DagNN.loadobj(net) ;
pDag = dag ;

dag.addLayer('softmax', dagnn.SoftMax(), dag.layers(end).outputs, 'prob', {}) ;

% comparison model
compPath = '~/data/models/matconvnet/imagenet-matconvnet-alex.mat' ;
mDag = dagnn.DagNN.loadobj(load(compPath)) ;
