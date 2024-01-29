function [Xobs, vi_var,vi_param] = vi_init(Xobs, vi_var,vi_param)
% Initialize Variational Inference Tensor Decomposition

if nargin <3
    vi_var = struct();
end

% Tensor Rank
R = vi_param.R;

% Dimension of the problem
Xdims = size(Xobs);

% Fit offset ?
fit_offset_dim = vi_param.fit_offset_dim;
fit_offset = sum(fit_offset_dim)>0;

%% Priors

% Priors CP: precision and mean
if not(isfield(vi_var,'CP_prior_mean'))
    vi_var.CP_prior_mean = ...
        cellfun(@(Z) zeros(Z,R) , num2cell(Xdims),'UniformOutput', false);
end

if not(isfield(vi_var,'CP_prior_precision'))
    vi_var.CP_prior_precision =  ...
        cellfun(@(Z) 0.01*repmat(reshape(eye(R),[1,R*R]),Z,1) , num2cell(Xdims),'UniformOutput', false);
end

% Prior Offset: precision and mean
if not(isfield(vi_var,'offset_prior_mean'))
    if sum(fit_offset_dim)==1
        vi_var.offset_prior_mean = zeros(Xdims(find(fit_offset_dim)),1);
    else
        vi_var.offset_prior_mean = zeros(Xdims(find(fit_offset_dim)));
    end
end

if not(isfield(vi_var,'offset_prior_precision'))
    if sum(fit_offset_dim)==1
        vi_var.offset_prior_precision = 0.00001*ones(Xdims(find(fit_offset_dim)),1);
    else
        vi_var.offset_prior_precision = 0.00001*ones(Xdims(find(fit_offset_dim)));
    end
end

% Precision prior
if not(isfield(vi_var,'prior_a_mode'))
    prior_a_mode = 100; vi_var.prior_a_mode = prior_a_mode;
end

if not(isfield(vi_var,'prior_b_mode'))
    prior_b_mode = 1; vi_var.prior_b_mode = prior_b_mode;
end


% Precision prior
if not(isfield(vi_var,'prior_a_shared'))
    prior_a_shared = 100; vi_var.prior_a_shared=prior_a_shared;
end

if not(isfield(vi_var,'prior_b_shared'))
    prior_b_shared = 1; vi_var.prior_b_shared=prior_b_shared;
end

%% Init variational distributions

if not(iscell(vi_param.observed_data))
    if  all(vi_param.observed_data(:)==1)
        observed_data = 1;
        observed_id = (1:numel(Xobs))';
        vi_param.observed_data = observed_data;
    else
        observed_data= vi_param.observed_data;
        observed_id = find(observed_data);
    end
end

% Init Offset Mean
if not(isfield(vi_var,'offset_mean'))
    vi_var.offset_mean = fit_offset*init_offsets(Xdims, fit_offset_dim);
    vi_var.offset_mean = observed_data.*vi_var.offset_mean;
end

% Init Offset Variance
if not(isfield(vi_var,'offset_variance'))
    vi_var.offset_variance = fit_offset*abs(init_offsets(Xdims, fit_offset_dim));
    vi_var.offset_variance = observed_data.*vi_var.offset_variance;
end

% Init CP Mean
if not(isfield(vi_var,'CP_mean'))
    vi_var.CP_mean = cellfun(@(Z) rand(Z,R) , num2cell(Xdims),'UniformOutput', false);
    %vi_var.CP_mean = cellfun(@(Z) 0.1*rand(Z,R) , num2cell(Xdims),'UniformOutput', false);
end

% Init CP Variance
if not(isfield(vi_var,'CP_variance'))
    vi_var.CP_variance = init_CP_variance(Xdims,R);
end

%% Store tensor 1st and 2nd moments

% Get reconstructed tensor 1st and second moment
AAt = get_AAt(vi_var.CP_mean,vi_var.CP_variance);
tensor_mean  = tensor_reconstruct(vi_var.CP_mean);
tensor2_mean = tensor_reconstruct(AAt);

% Save tensor moments
if not(isfield(vi_var,'tensor_mean'))
    vi_var.tensor_mean  = tensor_mean;
end

if not(isfield(vi_var,'tensor2_mean'))
    vi_var.tensor2_mean = tensor2_mean;
end
%% Init Shape

if not(isfield(vi_var,'shape'))
    vi_var.shape = mean(Xobs(observed_id(:)));
end

if not(isfield(vi_param,'shape_limit'))
    vi_param.shape_limit = 0;
    
end

%% Exploit Sparse-Block Structure of the dataset

if isfield(vi_param,'sparse')
    if strcmp(vi_param.sparse,'block-sparse')
        if isfield(vi_param,'indices_start_stop')
            [Xobs, vi_var,vi_param] = block_sparse_init(Xobs, vi_var,vi_param);
        else
            error('Blocks Start-Stop indices not provided ')
        end
        
    elseif strcmp(vi_param.sparse,'false')
        %
    else
        error('Sparsity mode not Implemented')
    end
else
    vi_param.sparse = 'false';
end

%% Init Latents

if not(isfield(vi_var,'latent_mean'))
    % Variational Update: Latent U
    vi_var = vi_update_latent(Xobs, vi_var,vi_param);
end


%% Use gpuArrays
if not(isfield(vi_param, 'use_gpu'))
    vi_param.use_gpu = 0;
elseif vi_param.use_gpu
    [Xobs,vi_param,vi_var] = gpu_init(Xobs,vi_param,vi_var);
end

end



function CP_variance = init_CP_variance(Xdims,R)
CP_variance  = cellfun(@(Z) ...
    repmat(reshape(eye(R),[1,R*R]),Z,1) , num2cell(Xdims),'UniformOutput', false);

for dimn=1:length(Xdims)
    for dimi=1:size(CP_variance{1,dimn},1)
        tmp = 0.1*randn(R,R);
        tmp = tmp'*tmp;
        CP_variance{1,dimn}(dimi,:) = tmp(:);
    end
end

end

function offsets = init_offsets(Xdims, fit_offset_dim)
% Init Offset

offsets_tmp = 0*rand(fit_offset_dim.*Xdims+not(fit_offset_dim));
offsets     = repmat(offsets_tmp, fit_offset_dim + not(fit_offset_dim).*Xdims);

end












