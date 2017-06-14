
for ii = 1:numel(mDag.layers)
  fprintf('----\n') ;
  fprintf('M - name: %s\n', mDag.layers(ii).name) ;
  fprintf('M - inputs: %s\n', mDag.layers(ii).inputs{1}) ;
  fprintf('M - outputs: %s\n', mDag.layers(ii).outputs{1}) ;
  fprintf('----\n') ;
  fprintf('P - name: %s\n', pDag.layers(ii).name) ;
  fprintf('P - nputs: %s\n', pDag.layers(ii).inputs{1}) ;
  fprintf('P - outputs: %s\n', pDag.layers(ii).outputs{1}) ;
end

mDag.conserveMemory = 0 ;
pDag.conserveMemory = 0 ;

imSz = mDag.meta.inputSize ;
im = single(imresize(imread('peppers.png'), imSz(1:2))) ;
imMean = [123, 117, 104] ; 
imMean = reshape(imMean, 1, 1, 3) ;
im = bsxfun(@minus, im, imMean) ;

imagesc(im / 255) ;
zs_dispFig ; 

pTarget = pDag.getVarIndex('prob') ;
mTarget = mDag.getVarIndex('prob') ;

mDag.eval({'input', im}) ;
pDag.eval({'data', im / 255}) ;

pPreds = squeeze(pDag.vars(pTarget).value) ;
mPreds = squeeze(mDag.vars(mTarget).value) ;

[mScore, mI] = max(mPreds) ;
title(sprintf('%s (%d), score %.3f',...
mDag.meta.classes.description{mI}, mI, mScore)) ;

[pScore, pI] = max(pPreds) ;
title(sprintf('%s (%d), score %.3f',...
mDag.meta.classes.description{pI}, pI, pScore)) ;
