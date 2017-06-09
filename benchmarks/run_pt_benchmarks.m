% do a single pass over the imagenet validation data
model = 'alexnet-pt-mcn' ;
[~,info] = cnn_imagenet_pt_mcn('model', model) ;
top1 = info.val.top1err * 100 ; top5 = info.val.top5err * 100 ;
fprintf('%s: top-1: %.2f, top-5: %.2f\n', model, top1, top5) ;
