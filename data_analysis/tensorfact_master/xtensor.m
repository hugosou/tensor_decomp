function xT = xtensor(varargin)
%XTENSOR - generalized product on tensors: X = xtensor(x1, dim1, x2, dim2, ...)
%
% X = XTENSOR(X1, DIM1, X2, DIM2) multiplies the arrays X1 and X2 according to
%    the generalised tensor product rules specified by DIMS1 and DIMS2.
%
%    If INTERSECT(DIMS1, DIMS2) == [], X is the tensor (outer) product of X1 and
%    X2.  The order of indices in X is given by the numeric values in DIMS1 and
%    DIMS2.
%
%    eg  if X1 is MxN and X2 is PxQxR,
%      X = XTENSOR(X1, [3,1], X2, [4,2,6]) is NxQxMxPx1xR and
%      X(n, q, m, p, 1, r) == X1(m, n)*X2(p, q, r)
%
%    If DIMS1 and DIMS2 share a positive index, the Schur (or .*) product is
%    taken along that dimension. 
%
%    eg  if X1 is MxP and X2 is NxP,
%      X = XTENSOR(X1, [1,3], X2, [2,3]) is MxNxP and
%      X(:, :, p) = X1(:, p)*X2(:, p)'
%
%    If DIMS1 and DIMS2 share a negative index, the tensor product is contracted
%    (summed) along the corresponding dimensions.  This corresponds to a matrix
%    product.
%
%    eg  if X1 is MxN and X2 is NxP,
%      XTENSOR(X1, [1,-1], X2, [-1,2]) == X1*X2
%
%    If DIMSn contains only negative integer entries (and is shorter
%    than NDIMS(Xn)) then each such index is taken to apply to the
%    dimension corresponding to its magnitude and the remaining
%    indices are preserved in order.
%
%    eg   if X1 is MxNxPxQ, X2 is NxNN and X3 is QxQQ then
%      XTENSOR(X1, [-2,-4], X2, [-2,2], X3, [-4,4]) == 
%      XTENSOR(X1, [1,-2,3,-4], X2, [-2,2], X3, [-4,4])
%
%    If DIMS1(J) == DIMS2(K) then either SIZE(X1,J) == SIZE(X2,K),
%    SIZE(X1,J) == 1 or SIZE(X2,K) == 1.  Where singleton dimensions imply
%    that the array should be copied a suitable number of times.
%
%    eg  if X1 is MxNxP...  and X2 is 1xNxP...,
%      XTENSOR(X1, [1:NDIMS(X1)], X2, [1:NDIMS(X2)]) == X1.*REPMAT(X2,[M,1,1,..])
%
% X = XTENSOR(X1, DIM1, X2, DIM2, X3, DIM3, ...) is similar to nested XTENSOR
%    calls, except in allowing the same negative (contraction) index to appear
%    more than twice, giving a non-standard mixture of Schur and contraction
%    products.
%
% X = XTENSOR({X1,X2,...}, {DIM1, DIM2, ...}) is equivalent to the
%    above forms.

% maneesh
% 20170714 - created
% 20170721 - added negative-only dimensions shortcut
% 20170722 - added cell form

% check that we have an even number of arguments
assert(rem(nargin, 2) == 0, ...
       'number of arguments must be even');

% check for cell inputs
if (iscell(varargin{1}))
  assert(nargin==2 && iscell(varargin{2}), ...
         'cell array syntax requires two cell arrays');
  TT = varargin{1};
  DD = varargin{2};
  assert(length(TT)==length(DD), ...
         'cell array arguments must have matched lengths');
else
  % extract the arguments into lists of tensors and dims
  TT = varargin(1:2:end);                 % tensors
  DD = varargin(2:2:end);                 % dimension lists
end

% check for negative-only dimension lists and inflate if needed
for ii = 1:length(DD)
  if all(DD{ii} < 0) && (length(DD{ii}) ~= ndims(TT{ii}))
    dimspec = 1:ndims(TT{ii});
    dimspec(-DD{ii}) = DD{ii};
    DD{ii} = dimspec;
  end
end

% collect all the referenced dimensions and check legality
DDall = unique(cat(2,DD{:}));
assertall( (DDall ~= 0) & (fix(DDall)==DDall), ...
       'Bad dimension index:', 'values', DDall, 'showIndices', 0);

DDmax = max(max(DDall),2);              % number of output dimensions - at least 2
DDtmp = [DDall(DDall < 0), 1:DDmax];    % fill out the positive dimensions

xT = 1;

for ii = 1:length(TT)
  assert(length(unique(DD{ii})) == length(DD{ii}), ...
         'Repeated dimension index for factor %d', ii);
  assert(max(length(DD{ii}),2) == ndims(TT{ii}), ...
         'Number of dimensions does not match for factor %d', ii);

  % dimensions to sum this time
  DDisum = setdiff(DD{ii}, cat(2, DD{ii+1:end}));
  DDisum = DDisum(DDisum < 0);

  % reorder to place these first
  DDitmp = [DDisum, setdiff(DDtmp, DDisum)];
  [~,iPerm] = ismember(DD{ii}, DDitmp);
  iPerm = [iPerm, setdiff(1:length(DDitmp), iPerm)];

  % permute current product the same way
  [~,pPerm] = ismember(DDtmp, DDitmp);
  
  % dotsum_ifneeded calls dotsum if there are .* products, otherwise uses permute+mtimes
  xT = dotsum_ifneeded(ipermute(xT, pPerm), ipermute(TT{ii}, iPerm), length(DDisum));
  
  % drop the contracted dimensions
  DDtmp = setdiff(DDtmp, DDisum);
  
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function zz = dotsum_ifneeded(xx,yy,nsum)
xsiz = size(xx);
ysiz = size(yy);
nd = max(numel(xsiz), numel(ysiz));     % max dimension
xsiz(end+1:nd) = 1;                     % pad with 1s
ysiz(end+1:nd) = 1;                     % pad with 1s

sxsiz = prod(xsiz(1:nsum));             % total summed dim size
sysiz = prod(ysiz(1:nsum));             % total summed dim size
xsiz(1:nsum) = [];                      % drop summed dims
ysiz(1:nsum) = [];                      % drop summed dims

if (sxsiz ~= sysiz) || any ((xsiz > 1) & (ysiz > 1))        % need Schur form?
  zz = dotsum(xx,yy,nsum);              %   yes - call dotsum
  
  %coder.ceval('dotsum.c',xx,yy,nsum)
else                                    %   no  - use mtimes
  xsingles = find(xsiz==1);             % singleton dims
  ysingles = find(ysiz==1 & xsiz > 1);  % singleton dims
  zz = reshape(xx, sxsiz, [])' * reshape(yy, sysiz, []);
  zz = reshape(zz, [xsiz(ysingles), ysiz(xsingles)]);
  zz = ipermute(zz, [ysingles, xsingles]);
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function xtensor_test()
  xx = randn(2, 2, 2, 2);
  yy = randn(2,2);
  zz = randn(2,2);
 
  %%
  assertall('eps', ...
            xtensor(xx,[1,2,3,4],yy,[3,4]) - ...
            bsxfun(@times, xx, permute(yy,[3,4,1,2])), ...
            {}, 'tolerance', 10*eps);
  %%
  assertall('eps', ...
            xtensor(xx,[-1,2,3,4]) - sum(xx,1), ...
            {}, 'tolerance', 10*eps);
  %%            
  assertall('eps', ...
            xtensor(xx,[1,2,-3,3],yy,[-3,3]) - ...
            squeeze(sum(bsxfun(@times, xx, permute(yy,[3,4,1,2])), 3)), ...
            {}, 'tolerance', 10*eps);
  %%
  assertall('eps', ...
            xtensor(xx,[1,2,-3,-4],yy,[-3,-4]) - ...
            squeeze(sum(bsxfun(@times, reshape(xx, [2,2,4]), reshape(yy,[1,1,4])), 3)), ...
            {}, 'tolerance', 10*eps);
  %%  
  assertall('eps', ...
            xtensor(xx,[1,2,-1,-2],yy,[-1,-2]) - ...
            xtensor(xx,[1,2,-3,-4],yy,[-3,-4]), ...
            {}, 'tolerance', 10*eps);
  %%
  assertall('eps', ...
            xtensor(xx,[-1,2,3,4],yy,[-1,1]) - ...
            reshape(permute(yy, [2,1]) * reshape(xx, 2, []), [2,2,2,2]), ...
            {}, 'tolerance', 10*eps);
  %%
  assertall('eps', ...
            xtensor(xx,[-1,1,-2,3],yy,[-1,1],yy,[-2,2]) - ...
            xtensor(xtensor(xx,[-1,1,2,3],yy,[-1,1]), [1,-2,3], yy, [-2,2]), ...
            {}, 'tolerance', 1*eps);
  %%
  assertall('eps', ...
            xtensor(xx,[-1,1,2,3],yy,[-1,1],zz,[-1,4]) - ...
            xtensor(xtensor(xx,[1,2,3,4],yy,[1,2]), [-1,1,2,3], zz, [-1,4]), ...
            {}, 'tolerance', 10*eps);
  
  %% speed tests
  xx = randn(100,100,100);
  yy = randn(100,100);
  tic, zz1 = xtensor(xx,[-1,1,3], yy,[-1,2]);   toc
  tic, zz2 = tensor(xx,[-1,1,3], yy,[-1,2]); toc
  tic, zz3 = permute(reshape(yy' * reshape(xx, 100, []), [100,100,100]), [2,1,3]); toc
  assertall('eps', zz1-zz2, {}, 'tolerance', 1e4*eps, 'showFails', 0);
  assertall('eps', zz1-zz3, {}, 'tolerance', 1e4*eps, 'showFails', 10);

  tic, zz4 = xtensor(xx,[-1,-2,3],yy,[-1,1],yy,[2,-2]); toc
  tic, zz5 = xtensor(xtensor(xx,[-1,2,3],yy,[-1,1]), [1,-2,3], yy, [2,-2]); toc
  tic, zz6 = tensor(tensor(xx,[-1,2,3],yy,[-1,1]), [1,-2,3], yy, [2,-2]); toc
  assertall('eps', zz4-zz5, {}, 'tolerance', 1e4*eps, 'showFails', 0);
  assertall('eps', zz4-zz6, {}, 'tolerance', 1e4*eps, 'showFails', 10);
end