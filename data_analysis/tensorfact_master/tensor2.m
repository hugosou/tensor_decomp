function xy = tensor(xx, xdims, yy, ydims)
%TENSOR - generalized product on tensors: xy = tensor(x, xdims, y, ydims)
%
% XY = TENSOR(X, XDIMS, Y, YDIMS) multiplies the arrays X and Y
%    according to the rules specified in XDIMS and YDIMS.
%
%    If INTERSECT(XDIMS, YDIMS) == [], XY is the tensor (outer)
%    product of X and Y.  The order of indices in XY is given by the
%    numeric values in XDIMS and YDIMS.
%
%    eg  if X is MxN and Y is PxQxR,
%      XY = TENSOR(X, [3,1], Y, [4,2,5]) is NxQxMxPxR and
%      XY(n, q, m, p, r) == X(m, n)*Y(p, q, r)
%
%    If XDIMS and YDIMS share a positive index, only the 'diagonal'
%    elements of the product are taken along that dimension.  This
%    corresponds to the .* operator in MATLAB.
%
%    eg  if X is MxP and Y is NxP,
%      XY = TENSOR(X, [1,3], Y, [2,3]) is MxNxP and
%      XY(:, :, p) = X(:, p)*Y(:, p)'
%
%    If XDIMS and YDIMS share a negative index, the tensor product
%    is contracted (summed) along the corresponding dimensions.  This
%    corresponds to a matrix product.
%
%    eg  if X is MxN and Y is NxP,
%      TENSOR(X, [1,-1], Y, [-1,2]) == X*Y
%
%    If XDIMS(J) == YDIMS(K) then either SIZE(X,J) == SIZE(Y,K),
%    SIZE(X,J) == 1 or SIZE(Y,K) == 1.  Where singleton dimensions imply
%    that the array should be copied a suitable number of times.
%
%    eg  if X is MxNxP...  and Y is 1xNxP...,
%      TENSOR(X, [1:NDIMS(X)], Y, [1:NDIMS(Y)]) == X.*REPMAT(Y,[M,1,1,..])

% convert logical data (or anything else) to double
if ~isa(xx, 'double') xx = double(xx); end
if ~isa(yy, 'double') yy = double(yy); end

xlen = length(xdims);		ylen = length(ydims);
xsize = size(xx);		ysize = size(yy);

if (xlen == 1 && all(xsize(2:end)==1))
  xlen = 2;
  xdims(2) = max([xdims(:);ydims(:)])+1;
end

if (ylen == 1 && all(ysize(2:end)==1))
  ylen = 2;
  ydims(2) = max([xdims(:);ydims(:)])+1;
end

if (xlen < ndims(xx) | ylen < ndims(yy))
  error ('indices don''t match dimensions');
end
xsize = ones(1,xlen);		ysize = ones(1,ylen);

xndim = xdims < 1;		yndim = ydims < 1;
xcont = xdims(xndim);		ycont = ydims(yndim);

if ~isempty(xcont) | ~isempty(ycont)
  if (sort(xcont) ~= sort(ycont))
    error('contraction index labels must be shared between X and Y');
  end
end

nsum = length(xcont);

[xcont xperm] = sort(xcont);	[ycont yperm] = sort(ycont);
xcont(xperm) = -nsum+1:0;	ycont(yperm) = -nsum+1:0;
xdims(xndim) = xcont;		ydims(yndim) = ycont;
xdims = xdims + nsum;		ydims = ydims + nsum;
mdim = max([xdims ydims]);

[xdims, xperm] = sort(xdims);	[ydims, yperm] = sort(ydims);
xsize(1:ndims(xx)) = size(xx);	ysize(1:ndims(yy)) = size(yy);
xtmpsize = ones(1, mdim);	ytmpsize = ones(1, mdim);
xtmpsize(xdims) = xsize(xperm);	ytmpsize(ydims) = ysize(yperm);

xy = dotsum(reshape(permute(xx, xperm),xtmpsize),...
            reshape(permute(yy, yperm),ytmpsize), nsum);
