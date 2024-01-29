function [Xobs, vi_var,vi_param] = block_sparse_init(Xobs, vi_var,vi_param)
% Keep only block tensors

Xdims = cellfun(@(Z) size(Z,1) ,vi_var.CP_mean);

% Observation
if not(iscell(Xobs))
    [Xobs, vi_param.indices_block] = block_sparse(Xobs, vi_param.indices_start_stop);
end

% Observed/Non observed data
if not(iscell(vi_param.observed_data))
    [vi_param.observed_data, ~] = block_sparse(vi_param.observed_data, vi_param.indices_start_stop);
end

% Offset mean
if not(iscell(vi_var.offset_mean))
    [vi_var.offset_mean, ~] = block_sparse(vi_var.offset_mean, vi_param.indices_start_stop);
end

% Offset variance
if not(iscell(vi_var.offset_variance))
    [vi_var.offset_variance, ~] = block_sparse(vi_var.offset_variance, vi_param.indices_start_stop);
end

% Reconstructed low rank Tensor 1st moment
if not(iscell(vi_var.tensor_mean))
    [vi_var.tensor_mean, ~]  = block_sparse(vi_var.tensor_mean , vi_param.indices_start_stop);
end
 
% Reconstructed low rank Tensor 2nd moment
if not(iscell(vi_var.tensor2_mean))
    [vi_var.tensor2_mean, ~] = block_sparse(vi_var.tensor2_mean, vi_param.indices_start_stop);
end

 
% Offset prior mean
if not(iscell(vi_var.offset_prior_mean))
    vi_var.offset_prior_mean = block_offset(vi_var.offset_prior_mean,...
        vi_param.fit_offset_dim, ...
        vi_param.indices_start_stop, Xdims);
end
 
% Offset prior precision
if not(iscell(vi_var.offset_prior_precision))
    vi_var.offset_prior_precision = block_offset(vi_var.offset_prior_precision,...
        vi_param.fit_offset_dim, ...
        vi_param.indices_start_stop, Xdims);
end


end