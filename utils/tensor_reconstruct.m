function X = tensor_reconstruct(A)
% Reconstruct a tensor from its CP-Decomposition A
% If X is a Tensor of size d1 x d2 x ... x dN
% A must be a cell array of size 1 x N
% A{1,i} is a matrix of size di x R
% R is the number of rank-1 tensor of the CP-Decomposition

Xdims = cellfun(@(Z) size(Z,1), A);
X = reshape(A{1}*KhatriRaoProd(A(end:-1:2))', Xdims);

end