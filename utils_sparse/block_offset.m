function prior_offset = block_offset(prior_offset, fit_offset_dim, block_indices_start_stop, Xdims)
% Break down the offset priors into blocks

% Dimension of the problems
alldims = 1:length(Xdims);

% Free and constrained variables
dim_vary = find(fit_offset_dim);
dim_cons = setdiff(alldims, dim_vary);

% Permutations Free /Constrained
perm_vary = [dim_vary, dim_cons];
perm_cons(perm_vary) = 1:length(perm_vary);

% Extend Compact Prior
prior_offset = repmat(prior_offset, [ones(1, length(dim_vary)), Xdims(dim_cons)]);
prior_offset = permute(prior_offset, perm_cons);

% Break it down in blocks
[prior_offset, ~]  = block_sparse(prior_offset, block_indices_start_stop);
 
% Recover compact form within blocks
num_blocks = size(prior_offset,1);
for block =1:num_blocks
    % Current block
    prior_mean_block = prior_offset{block} ;

    % Grasp Varying dimensions
    prior_mean_block = permute(prior_mean_block,perm_vary);
    block_dim = size(prior_mean_block);
    block_dim = block_dim(1:length(dim_vary));

    % Extract Varying dimensions
    prior_mean_block = prior_mean_block(1:prod(block_dim));
    
    if not(isempty(prior_mean_block))
        prior_mean_block = reshape(prior_mean_block, [block_dim,1]);

    else 
        % TODO: check this line ?
        prior_mean_block = zeros([block_dim,1]); 

    end

    % Store compact form
    prior_offset{block} = prior_mean_block;

end

end
