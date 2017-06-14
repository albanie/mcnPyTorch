function [y, dzdg, dzdb, moments] = vl_nnbnorm2(x, g, b, varargin) 

if ~isempty(varargin) && ~ischar(varargin{1})  % passed in dzdy
  dzdy = varargin{1} ;
  varargin(1) = [] ;
else
  dzdy = [] ;
end

opts.moments = [] ;
opts = vl_argparse(opts, varargin) ;

moments = [] ;

% first compute the statistics per channel
epsilon = 1e-4 ;
if isempty(opts.moments)
  mu = chanAvg(x) ;
  sigma2 = chanAvg(bsxfun(@minus, x, mu).^ 2) ;
  sigma = sqrt(sigma2 + epsilon) ;
else
  mu = permute(opts.moments(:,1), [3 2 1]) ;
  sigma = permute(opts.moments(:,2), [3 2 1]) ;
end

% normalize
x_hat = bsxfun(@rdivide, bsxfun(@minus, x, mu), sigma) ;

if isempty(dzdy)

	% apply gain
	res = bsxfun(@times, permute(g, [3 1 2]), x_hat) ;

	% add bias
	y = bsxfun(@plus, res, permute(b, [3 1 2])) ;

else
  % precompute some common terms 
  t1 = bsxfun(@minus, x, mu) ;
  t2 = bsxfun(@rdivide, 1, sqrt(sigma2 + epsilon)) ;
  t3 = -0.5 *(sigma2 + epsilon).^ (-3/2) ;
  sz = size(x) ; m = prod([sz(1:2) sz(4)]) ;

  dzdx_hat = bsxfun(@times, dzdy, permute(g, [3 1 2])) ;
  dzdsigma = chanSum(dzdx_hat .* bsxfun(@times, t1, t3)) ;

  m1 = chanSum(bsxfun(@times, dzdx_hat,  -1 * t2)) ;
  m2 = bsxfun(@times, -2*chanAvg(t1), dzdsigma) ;
  dzdmu = m1 + m2 ;

  dzdx = bsxfun(@times, dzdx_hat, t2) + ...
         bsxfun(@times, dzdsigma, 2 * t1 / m) ...
         + dzdmu * (1/m) ;
                                    
  y = dzdx ;
  dzdg = chanSum(dzdx_hat .* dzdy) ;
  dzdb = chanSum(dzdy) ;
end

% compute moments
if nargout == 2
		moments = horzcat(squeeze(mu), squeeze(sigma)) ;
    dzdg = moments ;
end

% ------------------------
function avg = chanAvg(x)
% ------------------------
avg = mean(mean(mean(x, 1), 2), 4) ;

% ------------------------
function res = chanSum(x)
% ------------------------
res = sum(sum(sum(x, 1), 2), 4) ;
