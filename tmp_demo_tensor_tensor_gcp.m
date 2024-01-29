%% GCP Tensor Decomposition of Count Data with constraints
addpath(genpath('./'))

%% Generate Dataset
% Model
add_offset  = 0;
add_missing = 0;
model_true  = 'negative_binomial';

% Observed Tensor Dimensions
Xdims = [100,70,3];

% True Rank
Rtrue = 2;

% For Reproduction purposes
rng(1)

% Simulate Toy Dataset
[Xobs,observed_data,true_params] = ...
    tmp_build_toydataset(model_true,Rtrue,Xdims,add_offset,add_missing);

% Grasp simulated parameters
param_names = fieldnames(true_params);
for parami = 1:length(param_names)
    eval([param_names{parami} '=true_params.' param_names{parami},';']);
end

% Plot True dataset 
plot_cp(true_params.CPtrue)


%% Fit Variational Inference
clc
R = 3;

% Fit parameters
vi_param = struct();
vi_param.ite_max = 4000;
vi_param.observed_data = observed_data;
vi_param.fit_offset_dim = add_offset*fit_offset_dim;
vi_param.shared_precision_dim= 1*[0,1,1];
vi_param.dim_neuron= 1;
vi_param.neurons_groups = neurons_groups;
vi_param.update_CP_dim = ones(1,ndims(Xobs));
vi_param.shape_update = 'MM-G';
vi_param.R = R;



vi_param.disppct = 1;


vi_var0 = struct();
vi_var0.shape = 120;

vi_var0.prior_a_mode = 10;
vi_var0.prior_b_mode = 1;

vi_var_with_ard = tensor_variational_inference(Xobs,vi_param,vi_var0);

vi_var_with_ardb = tensor_variational_inference(Xobs,vi_param,vi_var0);

%%

plot_cp(vi_var_with_ard.CP_mean)

%%

get_similarity({vi_var_with_ard.CP_mean, vi_var_with_ardb.CP_mean}, 1)



