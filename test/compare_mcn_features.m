function compare_mcn_features
% COMPARE_MCN_FEATURES - check mcn vs pytorch features
%
%   NOTE: Pytorch does not support "same" style padding. As a result,
%   it is difficult to reproduce features exactly that use zero padding
%   without a lot of extra computation (i.e. actively inserting
%   additional zeros into feature maps). If the scores are OK without
%   modifying the architecture, it's easier to leave it as it is.
%
% mcnPyTorch
% Licensed under The MIT License [see LICENSE.md for details]
% Copyright (C) 2017 Samuel Albanie

  opts.featurePath = 'data/mcnPyTorch/pyt-feats.mat' ;
  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts.modelPath = 'contrib/mcnPyTorch/models/resnet50-pt-mcn.mat' ;

  feats = load(opts.featurePath) ;
  dag = dagnn.DagNN.loadobj(load(opts.modelPath)) ;

  % reshape input image
  im = permute(feats.x0, [3 4 2 1]) ;
  data = im(:,:,[1 2 3]) ;

  dag.conserveMemory = 0 ;
  dag.mode = 'test' ;
  dag.eval({'data', data}) ;

  for ii = 1:numel(dag.vars) - 2
    xName = dag.vars(ii).name ;
    xName_ = sprintf('x%d', ii-1) ;
    x = dag.vars(ii).value ;
    x_ = permute(feats.(xName_), [3 4 2 1]) ;
    if contains(xName, 'conv')
      fprintf('size of %s vs %s\n', xName, xName_) ;
      fprintf('%s vs %s\n', mat2str(size(x)), mat2str(size(x_))) ;
      diff = sum(abs(x(:)) - abs(x_(:))) / sum(abs(x(:))) ;
      fprintf('diff: %g\n', diff) ;
    end
  end
