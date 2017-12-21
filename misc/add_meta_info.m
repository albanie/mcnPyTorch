function add_meta_info(varargin)
%ADD_META_INFO - adds additional meta information to imported models
% ADD_META_INFO adds informaiton about the imagenet dataset used for
% training to each model to facilitate easier use in deployment

  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts.imdbPath = fullfile(vl_rootnn, 'data/imagenet12/imdb.mat') ;
  opts = vl_argparse(opts, varargin) ;

  imdb = load(opts.imdbPath) ;

  res = dir(fullfile(opts.modelDir, '*resnet18-pt-mcn.mat')) ; modelNames = {res.name} ;
  for ii = 1:numel(modelNames)
    modelPath = fullfile(opts.modelDir, modelNames{ii}) ;
    fprintf('adding info to %s (%d/%d)\n', modelPath, ii, numel(modelNames)) ;
    net = load(modelPath) ;
    net.meta.classes = imdb.classes ;
    save(modelPath, '-struct', 'net') ;
  end

