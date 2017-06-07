function setup_mcnPyTorch()
%SETUP_MCNPYTORCH Sets up mcnPyTorch, by adding its folders to the Matlab path

  root = fileparts(mfilename('fullpath')) ;
  addpath(root) ;
  addpath(root, [root '/matlab'], [root '/num_diff']) ;
