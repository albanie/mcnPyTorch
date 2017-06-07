model = 'alexnet-mcn.mat' ;
modelDir = fullfile(vl_rootnn, 'contrib/mcnPyTorch/models') ;
net = load(fullfile(modelDir, model)) ;

dag = dagnn.DagNN.loadobj(net) ;
dag.addLayer('softmax', dagnn.SoftMax(), dag.layers(end).outputs, 'prob', {}) ;
