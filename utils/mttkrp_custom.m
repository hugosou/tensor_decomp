function MTTKRP = mttkrp_custom(tensor, factors, dimn, indices)
% Matrix tensor Khatri-Rao Product
% Adapted from Brett W. Bader, Tamara G. Kolda and others

% Dimension of the problem
Xdims = size(tensor);

% In case the last dimension is 1
if length(Xdims) < length(factors)
    Xdims = [Xdims, ones(1,length(factors)-length(Xdims))];
end

N = length(Xdims);
R = size(factors{1},2);
szl = prod(Xdims(1:dimn-1));
szr = prod(Xdims(dimn+1:N));
szn = Xdims(dimn);

if nargin <4
    do_reduce = 0;
else
    do_reduce = 1;
end
 
if dimn == 1
    % Fast to unfold 1st or last
    if do_reduce
        Ur = KhatriRaoProd(factors(N:-1:2), indices(N:-1:2));
    else
        Ur = KhatriRaoProd(factors(N:-1:2));
    end
    Y = reshape(tensor,szn,szr);
    MTTKRP =  Y * Ur;
    
elseif dimn == N
    % Fast to unfold 1st or last
    if do_reduce
        Ul = KhatriRaoProd(factors(N-1:-1:1), indices(N-1:-1:1));
    else
        Ul = KhatriRaoProd(factors(N-1:-1:1));
    end
    Y = reshape(tensor,szl,szn);
    MTTKRP = Y' * Ul;
    
else
    % Left and Right KhatriRao
    if do_reduce
        Ul = KhatriRaoProd(factors(N:-1:dimn+1),indices(N:-1:dimn+1));
        Ur = reshape(KhatriRaoProd(factors(dimn-1:-1:1),indices(dimn-1:-1:1)), szl, 1, R);
    else
        Ul = KhatriRaoProd(factors(N:-1:dimn+1));
        Ur = reshape(KhatriRaoProd(factors(dimn-1:-1:1)), szl, 1, R);
    end

    % Mult Left
    Y = reshape(tensor,[],szr);
    Y = Y * Ul;
    
    % Mult Right
    Y = reshape(Y,szl,szn,R);
    MTTKRP = bsxfun(@times,Ur,Y);
     
    % Reshape
    MTTKRP = reshape(sum(MTTKRP,1),szn,R);
end
end