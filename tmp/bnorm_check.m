eps = 1e-5 ;
%height = 20 ;
%width = 20 ;
%channels = 1024 ;

type = 'double' ;
modelName = 'resnext_101_64x4d' ;
p = load(fullfile(vl_rootnn, 'contrib/mcnPyTorch', ...
         sprintf('%s-params.mat', modelName))) ;
c = load(fullfile(vl_rootnn, 'contrib/mcnPyTorch', ...
         sprintf('%s-cat.mat', modelName))) ;

mult = p.('features_6_16_0_0_0_4_weight') ;
bias = p.('features_6_16_0_0_0_4_bias') ;
means = p.('features_6_16_0_0_0_4_running_mean') ;
var = p.('features_6_16_0_0_0_4_running_var') ;
x = permute(c.('x245'), [2 3 1]) ;

%sigma = sqrt((sqrt(var) + eps).^2 - eps) ;
moments = [means sqrt(var + eps)] ;
params = {mult, bias, moments} ;

res = vl_nnbnorm(x, mult, bias, 'moments', moments, 'epsilon', eps) ;
res2 = vl_nnbnorm2(x, mult, bias, 'moments', moments, 'epsilon', eps) ;

fprintf('bnorm out sum %g\n', sum(res(:))) ;
fprintf('bnorm2 out sum %g\n', sum(res2(:))) ;

