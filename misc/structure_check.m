
  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts.meanImg = [0.485, 0.456, 0.406] ;
  opts.std = [0.229, 0.224, 0.225] ;

  %modelName = 'resnext_50_32x4d-pt-mcn' ;
  modelName = 'inception_v3-pt-mcn' ;

  switch modelName
    case 'resnext_50_32x4d-pt-mcn', imsz = [224 224] ;
    case 'inception_v3-pt-mcn', imsz = [299 299] ;
  end

  if 1 % ~exist('dag', 'var') 
    modelPath = fullfile(opts.modelDir, sprintf('%s.mat', modelName)) ;
    net = load(modelPath) ;
    dag = dagnn.DagNN.loadobj(net) ;  
    dag.addLayer('softmax', dagnn.SoftMax(), dag.layers(end).outputs, 'prediction', {}) ;
  end

  path = '~/coding/libs/pretrained-models.pytorch/data/cat.jpg' ;
  im = single(imresize(imread(path), imsz)) ;
  data = im / 255 ; % scale to (almost) [0,1]
  data = bsxfun(@minus, data, permute(opts.meanImg, [1 3 2])) ;
  data = bsxfun(@rdivide, data, permute(opts.std, [1 3 2])) ;
  dag.mode = 'test' ;
  dag.move('gpu') ;

  data = gpuArray(data) ;
  dag.eval({'data', data}) ;

  % obtain the CNN otuput
  scores = dag.vars(end).value ;
  scores = squeeze(gather(scores)) ;

  % show the classification results
  [bestScore, best] = max(scores) ;
  imagesc(data) ; zs_dispFig ;
  predClass = dag.meta.classes.description{best} ;
  fprintf('top prediction: %s (conf: %.2f)\n', predClass, bestScore) ;
  dag.move('cpu') ;
