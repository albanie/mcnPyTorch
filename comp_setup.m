ref = load('~/data/models/matconvnet/imagenet-matconvnet-alex.mat') ;
labels = ref.meta.classes.description ;

%model = 'vgg11-mcn.mat' ;
model = 'alexnet-mcn.mat' ;
modelDir = fullfile(vl_rootnn, 'contrib/mcnPyTorch/models') ;
net = load(fullfile(modelDir, model)) ;

dag = dagnn.DagNN.loadobj(net) ;
dag.addLayer('softmax', dagnn.SoftMax(), dag.layers(end).outputs, 'prob', {}) ;

im = single(imresize(imread('peppers.png'), [227 227])) ;
im_ = im_ / 255 ;

imMean = [0.485 0.456 0.406] ;
std = [0.229 0.224 0.225] ;
im_ = bsxfun(@minus, im_, permute(imMean, [3 1 2])) ;
im_ = bsxfun(@rdivide, im_, permute(std, [3 1 2])) ;

dag.mode = 'test' ;
dag.eval({'data', im_}) ;

% obtain the CNN otuput
scores = dag.vars(dag.getVarIndex('prob')).value ;
scores = squeeze(gather(scores)) ;

[bestScore, best] = max(scores) ;
fprintf('%s (%d), score %.3f\n', labels{best}, best, bestScore) ;
