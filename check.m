
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

%target = pDag.getVarIndex('pred') ;
%pDag.vars(target).precious = 1 ;

mDag.eval({'input', im}) ;
pDag.eval({'data', im}) ;
%preds = dag.vars(target).value ;
