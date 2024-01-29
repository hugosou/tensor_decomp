function MTTKRP = mttkrp_block(block_tensor, block_indices , factors, dimn)
% Matrix tensor Khatri-Rao Product where tensor has block structure

D = size(factors{dimn},1);
R = size(factors{dimn},2);

num_blocks = length(block_tensor);

MTTKRP = zeros(D, R);
for block = 1:num_blocks
    
    MTTKRP(block_indices{block, dimn},:) = MTTKRP(block_indices{block, dimn},:) + ...
        mttkrp_custom(block_tensor{block}, factors, dimn, block_indices(block,:));
end
  
end

 