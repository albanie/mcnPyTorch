function setup_mcnPyTorch(varargin)
%SETUP_MCNPYTORCH Sets up mcnPyTorch, by adding its folders
% to the Matlab path
%
% Licensed under The MIT License [see LICENSE.md for details]
% Copyright (C) 2017 Samuel Albanie

  opts.dev = false ;
  opts = vl_argparse(opts, varargin) ;

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/test'], [root '/benchmarks']) ;
  addpath([root '/misc']) ;

  if opts.dev % only used for dev purposes
    addpath([root '/issue-fixes']) ;
  end
