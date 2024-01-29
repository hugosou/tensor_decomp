function block_tensor = tensor_reconstruct_block(factors, block_indices)
% Reconstruct a tensor from its CP-Decomposition A 
% block_tensor is a Tensor of size d1 x d2 x ... x dN
% A{1,i} is a matrix of size di x R
% X has a block structure defined in indices
% R is the number of rank-1 tensor of the CP-Decomposition

num_blocks = size(block_indices,1);

block_tensor = cell(num_blocks,1);
for block = 1:num_blocks
    
    block_index = block_indices(block,:);
    blovk_Xdims = cellfun(@(Z) length(Z) ,block_index);
    
    KRP = KhatriRaoProd(factors(end:-1:2),block_index(end:-1:2));
    block_tensor{block} = reshape(factors{1}(block_index{1},:)*KRP', blovk_Xdims);
    
end

end