function tensor_full = block_to_full(Xblock, Xbindex)
% Reconstruct a tensor from its block sparse representation


max_indices = cellfun(@(Z) max(Z(:)) ,Xbindex ,'UniformOutput' ,false);
is_empty = cellfun('isempty',max_indices);
max_indices(is_empty) = {0};  

tensor_full = zeros(max(cell2mat(max_indices)));

num_blocks = size(Xbindex,1);

for block = 1:num_blocks
    tensor_full(Xbindex{block, :}) = Xblock{block}; 
end

end
