function issue_6(varargin)

  opts.checkSample = false ;
  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts.checkPerf = true ;
  opts = vl_argparse(opts, varargin) ;


  if opts.checkSample
    % check code sample
    % Load network file
    modelPath = [opts.modelDir, '/resnet50-pt-mcn.mat'];
    net = dagnn.DagNN.loadobj(load(modelPath));

    % Allow extracting values from intermediate layers
    % https://github.com/vlfeat/matconvnet/issues/58
    net.conserveMemory = 0;

    sampleIm = single(imread('peppers.png')) ;
    I0 = sampleIm ;
    %I0 = single(my_read_image(i)); % Image in the 0..255 range

    % PyTorch normalization
    I1 = imresize(I0, net.meta.normalization.imageSize(1:2));
    I1 = I1 / 255 ; % scale to (almost) [0,1]
    I1 = bsxfun(@minus, I1, reshape(net.meta.normalization.averageImage, [1 1 3]));
    I = bsxfun(@rdivide, I1, reshape(net.meta.normalization.imageStd, [1 1 3]));

    % Feed the samples in batch
    net.eval({net.vars(1).name, I});

    % Extract features
    assert(length(net.vars) == length(net.layers)+1);
    N_LAYERS = length(net.layers) + 1;
    % validate forward pass
    %for l=1:N_LAYERS
    %    s = prod(layer_sizes{l+1});
    %    data = reshape(net.vars(l+1).value, [s, N_SAMPLES])';
    %        % ...further use data...
    %end
  end

  if opts.checkPerf
    run_pt_benchmarks('importedModels', {'resnet50-pt-mcn'}) ;
  end
