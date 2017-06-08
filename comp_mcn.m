net = load('~/data/models/matconvnet/imagenet-matconvnet-alex.mat') ;
labels = net.meta.classes.description ;
dag = dagnn.DagNN.loadobj(net) ;

im_ = single(imresize(imread('peppers.png'), dag.meta.normalization.imageSize(1:2))) ;
im_ = bsxfun(@minus, im_, permute(dag.meta.normalization.averageImage, [3 2 1])) ;

%imMean = [0.485 0.456 0.406] ;
%std = [0.229 0.224 0.225] ;
%im_ = bsxfun(@minus, im_, permute(imMean, [3 1 2])) ;
%im_ = bsxfun(@rdivide, im_, permute(std, [3 1 2])) ;

dag.mode = 'test' ;
dag.eval({'input', im_}) ;

% obtain the CNN otuput
scores = dag.vars(dag.getVarIndex('prob')).value ;
scores = squeeze(gather(scores)) ;

[bestScore, best] = max(scores) ;
fprintf('%s (%d), score %.3f\n', labels{best}, best, bestScore) ;
