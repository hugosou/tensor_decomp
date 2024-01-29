%% Probabilistic Tensor Decomposition of Count Data
% Illustrate ARD for tensor rank (1/2): Missing Data + Offset
addpath(genpath('./'))
addpath(genpath('./../tensor_decomp/'))
% 
%% Generate Dataset
% Model
add_offset  = 1;
add_missing = 1;
model_true  = 'negative_binomial';

% Observed Tensor Dimensions
Xdims = [100,70,3,4,5];

% True Rank
Rtrue = 4;

% For Reproduction purposes
rng(1)

% Simulate Toy Dataset
[Xobs,observed_data,true_params] = ...
    build_toydataset(model_true,Rtrue,Xdims,add_offset,add_missing);

% Grasp simulated parameters
param_names = fieldnames(true_params);
for parami = 1:length(param_names)
    eval([param_names{parami} '=true_params.' param_names{parami},';']);
end

% Plot True dataset
%plot_cp(true_params.CPtrue)

%% Variational Inference
clc
R = 16;


% Fit parameters
vi_param = struct();
vi_param.ite_max = 4000;
vi_param.observed_data = observed_data;
vi_param.fit_offset_dim = add_offset*fit_offset_dim;
vi_param.shared_precision_dim= 1*[0,1,1,1,1];
vi_param.dim_neuron= 1;
vi_param.neurons_groups = neurons_groups;
vi_param.update_CP_dim = ones(1,ndims(Xobs));
vi_param.shape_update = 'MM-G';
vi_param.R = R;

% Shared initialization
vi_var = struct();
vi_var.shape = 120;
vi_param.disppct = 0.1;


if add_missing
    od = squeeze(observed_data(:,1,1,:,1));
    %figure; imagesc(od)
    
    id_bocks_d1 = find_x_blocks(od);
    id_bocks_d4 = find_x_blocks(od');
    
    num_blocks = size(id_bocks_d1,1);
    assert(num_blocks == size(id_bocks_d4,1))
    
    
    id_bocks_d2 = repmat([1,size(Xobs,2)],num_blocks,1);
    id_bocks_d3 = repmat([1,size(Xobs,3)],num_blocks,1);
    id_bocks_d5 = repmat([1,size(Xobs,5)],num_blocks,1);
    
    Xbindex_ste = [...
        mat2cell(id_bocks_d1, ones(1,num_blocks)),...
        mat2cell(id_bocks_d2, ones(1,num_blocks)),...
        mat2cell(id_bocks_d3, ones(1,num_blocks)),...
        mat2cell(id_bocks_d4, ones(1,num_blocks)),...
        mat2cell(id_bocks_d5, ones(1,num_blocks))];
    
    [Xblock, Xbindex] = block_sparse(Xobs,Xbindex_ste);
    [Oblock, ~] = block_sparse(observed_data,Xbindex_ste);
    
    viparam0 = vi_param;
    viparam0.shape_limit = 100000;
    
    viparam0.sparse = 'block-sparse';
    viparam0 = Xbindex_ste;
    vi_var0 = tensor_variational_inference(Xobs,viparam0, vi_var);
    
else
    viparam1 = vi_param;
    viparam1.sparse = 'false';
    vi_var1 = tensor_variational_inference(Xobs,viparam1, vi_var);
    
end

% 
% % 
% % %% ALSO BROKEN WHEN NO MISSING DATA !
% %  
% % % Error CP
% % error_cp = zeros(ndims(Xobs),1);
% % for dimn=1:ndims(Xobs)
% %     error_cp(dimn) = sum(sum(abs(vi_var0.CP_mean{dimn}-vi_var1.CP_mean{dimn})));
% % end
% % disp(['Error factors = ', num2str(max(error_cp)')])
% % 
% % % Error latent
% % U1 = vi_var1.latent_mean.*viparam1.observed_data;
% % U2 =  block_to_full(vi_var0.latent_mean, Xbindex).*viparam1.observed_data;
% % 
% % % Error Offset
% % O1 = vi_var1.offset_mean.*viparam1.observed_data;
% % O2 =  block_to_full(vi_var0.offset_mean, Xbindex);
% % 
% % P1 = sum(abs([vi_var0.a_shared; vi_var0.b_shared]...
% %    -[vi_var1.a_shared; vi_var1.b_shared]));
% % 
% % P2 = sum(abs([vi_var0.a_mode; vi_var0.b_mode]...
% %    -[vi_var1.a_mode; vi_var1.b_mode]));
% % 
% % disp(['Error factors = ', num2str(max(error_cp)')])
% % disp(['Error latents = ', num2str(max(abs(U1(:)-U2(:))))])
% % disp(['Error offsets = ', num2str(max(O1(:)-O2(:)))])
% % disp(['Error shared  = ', num2str(max(P1))])
% % disp(['Error modes   = ', num2str(max(P2))])
% % disp(['Error shape   = ', num2str(abs(vi_var0.shape-vi_var1.shape))])
% %  
% % 
% % 
% % 
% % %%
% % vi_var0 = vi_var;
% % vi_var1 = vi_var;
% % 
% % 
% % 
% % 
% % vi_var0 = vi_update_CP_old(vi_var0,vi_param, Xobs);
% % vi_var1 = vi_update_CP(Xobs, vi_var1,vi_param);
% % 
% % %vi_var0 = vi_update_latent_old(Xobs0, vi_var0,viparam0);
% % %vi_var1 = vi_update_latent(Xobs1, vi_var1,viparam1);
% % 
% % %vi_var0 = vi_update_offset_old(Xobs0, vi_var0,viparam0);
% % %vi_var1 = vi_update_offset(Xobs1, vi_var1,viparam1);
% % 
% % %vi_var0 = vi_update_precision_shared(vi_var0,viparam0);
% % %vi_var1 = vi_update_precision_shared(vi_var1,viparam1);
% % 
% % %vi_var0 = vi_update_precision_mode(vi_var0,viparam0);
% % %vi_var1 = vi_update_precision_mode(vi_var1,viparam1);
% % 
% % %vi_var1 = vi_update_shape_old(Xobs1, vi_var1,viparam1);
% % %vi_var0 = vi_update_shape(Xobs0, vi_var0,viparam0);
% % 
% % 
% % 
% % %% SHAPE UPDATE IS BROKEN !!
% % %% ALSO BROKEN WHEN NO MISSING DATA !
% % rng(1)
% % [Xobs, vi_var,vi_param] = vi_init(Xobs, vi_var, vi_param1);
% % 
% % 
% % 
% % %%
% % 
% % 
% % if add_missing
% %     vi_param.sparse = 'block-sparse';
% %     vi_param.indices_start_stop = Xbindex_ste;
% % else
% %     vi_param.sparse = 'false';
% % end
% % 
% % %%
% % 
% % 
% % 
% % %%
% % 
% % %%
% % % 
% % % observed_data = blkdiag(ones(200,200),ones(100,300),ones(800,100),ones(100,150),ones(100,150));
% % % D1 = size(observed_data ,1);
% % % D2 = 70;
% % % D3 = size(observed_data ,2);
% % % 
% % % observed_data = repmat(observed_data, [1,1, D2]);
% % % observed_data = permute(observed_data, [1,3,2]);
% % % 
% % % 
% % % Xobs = rand(D1,D2,D3);
% % % Xobs = Xobs.*observed_data;
% % % 
% % % R = 15*15;
% % % sizes1 = mat2cell(size(Xobs),1,ones(1,ndims(Xobs)));
% % % sizes2 = mat2cell(repmat(R,1,ndims(Xobs)),1,ones(1,ndims(Xobs)));
% % % factors= cellfun(@(X,Y) rand(X,Y), sizes1, sizes2, 'UniformOutput', false);
% % % 
% % % %%
% % % od = squeeze(observed_data(:,1,:));
% % % figure; imagesc(od)
% % % 
% % % id_bocks_d1 = find_x_blocks(od);
% % % id_bocks_d3 = find_x_blocks(od');
% % % num_blocks = size(id_bocks_d1,1);
% % % assert(num_blocks == size(id_bocks_d3,1))
% % % id_bocks_d2 = repmat([1,D2],num_blocks,1);
% % % 
% % % 
% % % Xbindex_ste = [...
% % %     mat2cell(id_bocks_d1, ones(1,num_blocks)),...
% % %     mat2cell(id_bocks_d2, ones(1,num_blocks)),...
% % %     mat2cell(id_bocks_d3, ones(1,num_blocks))];
% % % 
% % % 
% % % 
% % % 
% % 
% % 
% % 
% % 
% % vi_var0 = vi_update_CP_old(vi_var0,vi_param, Xobs);
% % vi_var1 = vi_update_CP(Xobs, vi_var1,vi_param);
% % 
% % %vi_var0 = vi_update_latent_old(Xobs0, vi_var0,viparam0);
% % %vi_var1 = vi_update_latent(Xobs1, vi_var1,viparam1);
% % 
% % %vi_var0 = vi_update_offset_old(Xobs0, vi_var0,viparam0);
% % %vi_var1 = vi_update_offset(Xobs1, vi_var1,viparam1);
% % 
% % %vi_var0 = vi_update_precision_shared(vi_var0,viparam0);
% % %vi_var1 = vi_update_precision_shared(vi_var1,viparam1);
% % 
% % %vi_var0 = vi_update_precision_mode(vi_var0,viparam0);
% % %vi_var1 = vi_update_precision_mode(vi_var1,viparam1);
% % 
% % %vi_var1 = vi_update_shape_old(Xobs1, vi_var1,viparam1);
% % %vi_var0 = vi_update_shape(Xobs0, vi_var0,viparam0);
% % 
% % 
% % %%
% % 
% % % Error CP
% % error_cp = zeros(ndims(Xobs),1);
% % for dimn=1:ndims(Xobs)
% %     error_cp(dimn) = sum(sum(abs(vi_var0.CP_mean{dimn}-vi_var1.CP_mean{dimn})));
% % end
% % disp(['Error factors = ', num2str(max(error_cp)')])
% % 
% % 
% % %%
% %  
% % % Error latent
% % U1 = vi_var1.latent_mean.*viparam1.observed_data;
% % U2 =  block_to_full(vi_var0.latent_mean, viparam0.indices_block).*viparam1.observed_data;
% % 
% % % Error Offset
% % O1 = vi_var1.offset_mean.*viparam1.observed_data;
% % O2 =  block_to_full(vi_var0.offset_mean, viparam0.indices_block);
% % 
% % 
% % P1 = sum(abs([vi_var0.prior_a_shared; vi_var0.prior_b_shared]...
% %     -[vi_var1.prior_a_shared; vi_var1.prior_b_shared]));
% % 
% % P2 = sum(abs([vi_var0.a_mode; vi_var0.b_mode]...
% %     -[vi_var1.a_mode; vi_var1.b_mode]));
% % 
% % disp(['Error factors = ', num2str(max(error_cp)')])
% % disp(['Error latents = ', num2str(max(U1(:)-U2(:)))])
% % disp(['Error offsets = ', num2str(max(O1(:)-O2(:)))])
% % disp(['Error shared  = ', num2str(max(P2))])
% % disp(['Error modes   = ', num2str(max(P1))])
% % disp(['Error shape   = ', num2str(abs(vi_var0.shape-vi_var1.shape))])
% %  
% % 
% % 
% % 
% % %%
% % 
% % 
% % 
% % 
% % 
% % tic
% % for dimn =2
% %     Y = mttkrp_block(Xbobs, factors, dimn);
% % end
% % toc
% % 
% % tic
% % for dimn =2
% %     Y2 = mttkrp_custom(Xobs, factors, dimn);
% % end
% % toc
% % 
% % 
% % %%
% % Xdims = size(Xobs);
% % tic
% % X1 = tensor_reconstruct(factors);
% % toc
% % 
% % tic
% % Xb2 =  tensor_reconstruct_block(factors, Xbindex);
% % toc
% % 
% % 
% % %%
% % XX= block_to_full(Xb2, Xbindex);
% % 
% % X1 = X1.*observed_data;
% % 
% % sum(abs(X1(:)-XX(:)))
% % % 


%%

% Which dataset to use: 'mismatch', 'standard'
type_data = 'standard';
%type_data = 'mismatch';

% Data Analysis Mode
mode = 'distinguish_rotation';
%mode = 'stitch';

% Machine on which to load data
%machine = 'cluster';
machine = 'local';

if strcmp(mode, 'distinguish_rotation')
    % Dataset Parameters
    flip_back_data = 1;
    distinguish_rotation = flip_back_data;
    kept_trials_num = 6;
    
elseif strcmp(mode, 'stitch')
    % Dataset Parameters
    flip_back_data = 0;
    kept_trials_num = 2*7;
    distinguish_rotation = flip_back_data;
else
    error('Not Implemented Mode')
end

% Discard recordings with not enough trials
discard_low =1;

% Split data for cross validation or train on full data
train_test_split = 'split';
%train_test_split = 'train';

% Iterations
ite_max = 10000;

% Maximum number of xval folder
k_max = min(nchoosek(kept_trials_num,kept_trials_num/2), 24);


% Keep regions from : 'N/A','RSPd','RSPg','SC','SUB','V1', 'Hip'
if strcmp(type_data,'standard')
    kept_region = {'N/A','RSPd','RSPg','SC','SUB','V1', 'Hip'};
elseif strcmp(type_data,'mismatch')
    kept_region = {'RSPd','RSPg','SC'};
else
    error('Incorect type data')
end

% Add master folders
addpath(genpath('~/Documents/MATLAB/tensor_decomp/data_analysis/'))
addpath(genpath('~/Documents/MATLAB/tensor_decomp/utils/'))
addpath(genpath('~/Documents/MATLAB/tensor_decomp/tensor_vi/'))

% Input / Output Files
if strcmp(machine,'cluster')
    data_folder = '/nfs/gatsbystor/hugos/data_sepi_full_trial/';
    resu_folder = '/nfs/gatsbystor/hugos/data_sepi_full_trial/';
elseif strcmp(machine,'local')
    data_folder = '~/Documents/Data/data_sepi_full_trial/';
    resu_folder = '~/Documents/Data/data_sepi_full_trial/';
else
    error('Incorect machine')
end


% Load Pre-processed Data
name_data = [data_folder,'data_sepi_',type_data, '_full_trial_flipback', num2str(flip_back_data)];
data = load(name_data);
datam = data.data_sepi;

% Attribute to group neurons: 'ctype','layer','region'
neuron_group_discriminant = 'layer';

% Reorder Data
data_sepi = data_load_and_align(...
    datam, kept_trials_num, ...
    kept_region, neuron_group_discriminant, ...
    discard_low, distinguish_rotation);

% Extract experimental parameters
param_names = fieldnames(data_sepi);
for parami = 1:length(param_names)
    eval([param_names{parami} '=data_sepi.' param_names{parami},';']);
end

experimental_parameters = data_sepi;
experimental_parameters.type_data=type_data;
experimental_parameters.distinguish_rotation = distinguish_rotation;
experimental_parameters.mode = mode;

observed_tensor = experimental_parameters.observed_tensor;
observed_data   = experimental_parameters.observed_data;
observed_dims   = size(observed_tensor);

if strcmp(machine, 'local')
    
    %data_observed_old = permute(squeeze(sum(datam.observed_data(:,1,:,:,:),4)),[2,1,3]);
    %data_observed_new = permute(squeeze(sum(data_sepi.observed_data(:,1,:,:,:),4)),[2,1,3]);
    
    %figure
    %plot_before_after(data_observed_old, data_observed_new,...
    %    [[0,0,0];[0, 1, 0]], [0,1], datam.condition, 'Observed (0: black 1: Green)')
    %set(gcf, 'position', [ 2042         497         671         397]);
    
    data_direction_old = datam.direction;
    data_direction_new = data_sepi.direction;
    
    figure
    plot_before_after(data_direction_old, data_direction_new,...
        [[246, 161, 41]/255; [0,0,0]; [106, 191, 187]/255], [-1,1], datam.condition, 'CCW - CW - N/A')
    set(gcf, 'position', [ 2042         497         671         397]);
end



%%


Xdims = size(observed_tensor);

Xobs = reshape(permute(observed_tensor, [1,2,3,5,6,4]), Xdims(1),Xdims(2), []);
Oobs = reshape(permute(observed_data, [1,2,3,5,6,4]), Xdims(1),Xdims(2), []);
%observed_data = reshape(permute(observed_data, [1,2,3,5,4]), Xdims(1),Xdims(2), []);

figure
imagesc(squeeze(Oobs(:,1,:)))

R = 15*15;
sizes1 = mat2cell(size(Xobs),1,ones(1,ndims(Xobs)));
sizes2 = mat2cell(repmat(R,1,ndims(Xobs)),1,ones(1,ndims(Xobs)));
factors= cellfun(@(X,Y) rand(X,Y), sizes1, sizes2, 'UniformOutput', false);

od = squeeze(Oobs(:,1,:));
figure; imagesc(od)

id_bocks_d1 = find_x_blocks(od);
id_bocks_d3 = find_x_blocks(od');
num_blocks = size(id_bocks_d1,1);
assert(num_blocks == size(id_bocks_d3,1))
id_bocks_d2 = repmat([1,size(Xobs,2)],num_blocks,1);



Xbindex_ste = [...
    mat2cell(id_bocks_d1, ones(1,num_blocks)),...
    mat2cell(id_bocks_d2, ones(1,num_blocks)),...
    mat2cell(id_bocks_d3, ones(1,num_blocks))];


%%

clc
R = 15;


% Fit parameters
vi_param = struct();
vi_param.ite_max = 10;
vi_param.observed_data = Oobs;
vi_param.fit_offset_dim = [1,0,1];
vi_param.shared_precision_dim= [0,1,1];
vi_param.dim_neuron= 1;
vi_param.neurons_groups = experimental_parameters.neuron_group{1};
vi_param.shape_update = 'MM-G';
vi_param.R = R;

% Shared initialization
vi_var = struct();
vi_var.shape = 120;
vi_param.disppct = 0.1;


add_missing =1;

if add_missing
    viparam0 = vi_param;
    viparam0.sparse = 'block-sparse';
    viparam0.shape_limit = 100000;
    viparam0.indices_start_stop = Xbindex_ste;
    tic
    vi_var0 = tensor_variational_inference(Xobs,viparam0, vi_var);
    toc
    
else
    viparam1 = vi_param;
    viparam1.sparse = 'false';
    tic
    vi_var1 = tensor_variational_inference(Xobs,viparam1, vi_var);
    toc
end




function plot_before_after(data1, data2, colormapx, scale, titles1, titles2)

for condi =1:size(data1,1)
    subplot(2,size(data1,1),condi)
    imagesc(squeeze(data1(condi,:,:)))
    title(titles1{condi})
    xlabel('Trial')
    ylabel('Full Session')
    caxis(scale)
    colormap(colormapx);
end


for condi =1:size(data2,1)
    subplot(2,size(data2,1),condi + size(data2,1))
    imagesc(squeeze(data2(condi,:,:)))
    title(titles2)
    xlabel('Trial')
    ylabel('Kept Session')
    caxis(scale)
    colormap(colormapx);
end

end

function tensor_new = collapse_trials(tensor, folders, mode)
    if strcmp(mode,'stitch')
        tensor_new = stich_and_collapse_trial(tensor, folders);
    elseif strcmp(mode,'distinguish_rotation')
        tensor_new = split_and_collapse_trial(tensor, folders);
    else
        error('Not Implemented')
    end
end

function tensor_new = split_and_collapse_trial(tensor, folders)
    tensor_new = sum(tensor(:,:,:,:,folders,:), [4, 5]);
    tensor_new = tensor_new(:,:,:);
end

function tensor_new = stich_and_collapse_trial(tensor, folders)
    tensor_new = sum(tensor(:,:,:,:,folders,:), [4]);
    tensor_new = tensor_new(:,:,:);
end

