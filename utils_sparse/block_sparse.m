function [Xblocks, Xbindex] = block_sparse(tensor,start_stop_indices)

num_blocks = size(start_stop_indices,1);

Xbindex = cellfun(@(Z) Z(1):Z(end), start_stop_indices, 'UniformOutput', false);

Xblocks = cell(num_blocks, 1);

if not(isempty(tensor))
    for block = 1:num_blocks
        Xblocks{block} = tensor(Xbindex{block,:});
    end
end

end


