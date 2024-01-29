function [UU,GG,varexp] = tnsrfactregress(YY,XX,KK, varargin)
% tnsrfactregress - tensor-factored (low-rank) regression
%
% [U,G] = TNSRFACTREGRESS(Y,X,K) where Y is MxNx..xT and X is SxT
%    performs (multilinear) low-rank factored least-squares regression
%    from the columns of X to the MxNx... tensors stacked in Y.  Let D
%    = ndims(Y).  K is the multilinear rank and should be vector of
%    length D+1. The regression weights are returned as a cell array
%    of (D+1) orthogonal basis vectors U, and a core tensor G so that
%    size(G) = K and size(U{d}) = [size(Y,d) K(d)] for d = 1 to D,       
%    while size(U{D+1}) = [S, K(D+1)].  The combined regression weights 
%    are given by:
%    W = xtensor(G, [-1,-2,...], U{1}, [1,-1], U{2}, [2, -2], ...)
%
%    The rank trunction is performed by (S)T-HOSVD and so is approximate.  
%    Note that the columns of each U{d} and corresponding slices of
%    G are ordered in terms of variance contributed (in the sense of 
%    (S)T-HOSVD).
%
%    This is a solution to 'tensor low-rank regression' that,
%    unlike the HOLRR of Rabusseau and Kadri, remains effective
%    when the noise in Y is not white, when the rank is reduced
%    beyond that of the true relationship, or when the true relationship 
%    is non-linear. 

% maneesh. 
% 20170701: created 
% 20170712: test code and compare to HOLRR algorithm


sequence = [];
seqtrunc = 1;
optlistassign(who, varargin);


Nyy = size(YY);
Nt = Nyy(end); Nyy(end) = [];  
Dy = length(Nyy);

assert(ndims(XX) == 2, 'X must be a 2-way array');
[Nx,Ntx] = size(XX);


assert(Nt == Ntx, 'input dimension mismatch');
assert(Dy+1 == length(KK), 'rank specification does not match dimensions');

if isempty(sequence)
  % use a variant on the VVM order heuristic for the output dimensions
  [~,sequence] = sort(Nyy - KK(1:Dy), 2, 'descend');
else
  if length(sequence) == 1 && sequence == 0
    sequence = 1:Dy;
    seqtrunc = 0;
  end
end



XX2 = XX*XX';

% full rank regression 
WW = reshape(YY, prod(Nyy), Nt)*XX' / XX2;


%XX2
%WW'*WW*

% reduce input rank
[UUx, varexp] = eigs(WW'*WW*XX2, KK(end));   % usually WW'*YY*XX', but unpredicted YY is in null of WW
[UUx, ~,~] = svd(UUx, 0);               % orthogonalise
UU{Dy+1} = UUx;                         % store

% (S)T-HOSVD on *predictions*
% S-version: Vannieuwenhoven, Vandebril & Meerbergen, 2012, SIAM J Sci Comput (VVM)
% otherwise, use standard truncated HOSVD

if (seqtrunc)
  YY0 = reshape(WW*UUx*UUx'*XX, [Nyy, Nt]);% current truncated prediction
else
  YY0 = reshape(WW*XX, [Nyy, Nt]);      % prediction
end  


KK0 = Nyy;                            % current truncated dims
WW = reshape(WW, [Nyy, Nx]);            % reshape weights



for dd = sequence
  ddperm = [dd, setdiff(1:Dy, dd), Dy+1]; % pull current dim to front
  YYd = reshape(permute(YY0, ddperm), KK0(dd), []);
  [UU{dd},~] = eigs(YYd*YYd', KK(dd));
  if seqtrunc
    KK0(dd) = KK(dd);
    YY0 = ipermute(reshape(UU{dd}'*YYd, [KK0(ddperm(1:end-1)), Nt]), ddperm);
  end
end

%cat(2, {-[1:Dy+1]},  cellfun(@(x)[-x,x], num2cell(1:Dy+1), 'UniformOutput', 0))

GG = xtensor({WW, UU{:}}, cat(2, {-[1:Dy+1]},  cellfun(@(x)[-x,x], num2cell(1:Dy+1), 'UniformOutput', 0)));

                      

%% use standard RRR to reduce the input rank
%YYd = reshape(YY0, [], Nt);
%[UU0, ~] = eigs(YYd*YYd', KK(Dy+1));
%[UU0,GG0,UU{Dy+1}] = svd(UU0'*reshape(WW0, [], Nx));
%GG = reshape(UU0*GG0, KK);




% return variance explained if requested
%if (nargout > 2)
%  varargout{1} = diag(sv).^2;
%end


