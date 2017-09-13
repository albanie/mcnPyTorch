function setup_mcnPyTorch()
%SETUP_MCNPYTORCH Sets up mcnPyTorch, by adding its folders 
% to the Matlab path
%
% Licensed under The MIT License [see LICENSE.md for details]
% Copyright (C) 2017 Samuel Albanie 

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/test'], [root '/benchmarks']) ;
  addpath([root '/misc']) ;
