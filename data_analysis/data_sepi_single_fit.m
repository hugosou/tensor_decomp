%% Load and organize data

% Add master folders
%addpath(genpath('~/Documents/MATLAB/tensor_decomp2/'))

addpath(genpath('~/Documents/MATLAB/'))


data_folder = '/nfs/gatsbystor/hugos/data_sepi_all/';
resu_folder = '/nfs/gatsbystor/hugos/';

%data_folder = '~/Documents/Data/data_sepi_all/';
%resu_folder = '~/Documents/Data/data_sepi_all/';

% Dataset Parameters
flip_back_data = 0;
stitch_data    = 1;

% Which dataset to use: 'mismatch', 'standard'
type_data = 'standard';

% Load
name_data = [data_folder,'data_sepi_',type_data, '_stitching', num2str(stitch_data), '_flipback', num2str(flip_back_data)];
load(name_data)

% Keep regions from : 'N/A','RSPd','RSPg','SC','SUB','V1', 'Hip'
kept_region = {'RSPd','RSPg','SC','SUB','V1'};

% Attribute to group neurons: 'ctype','layer','region'
neuron_group_discriminant = 'layer';

% Load
data_sepi = data_sepi_load(data_sepi,kept_region,neuron_group_discriminant);

% Extract experimental parameters
param_names = fieldnames(data_sepi);
for parami = 1:length(param_names)
    eval([param_names{parami} '=data_sepi.' param_names{parami},';']);
end

Xobs = observed_tensor;

% Use a 3D or 4D tensor
collapse_conditions = 1;

% observed_tensor dim
if collapse_conditions
    final_dim = 3;
else 
    final_dim = ndims(Xobs);
end
experimental_parameters = data_sepi;
%% Fit Parameters

% Neuron-type wise constraint
Pg = neuron_group;

% Iterations and gradient steps
ite_max = 2000; 
rho_max = 1e-8;
rho_min = 5e-1;
tau     = 1000;
period  = ite_max/200; 
etaf    = get_step(ite_max, rho_max,rho_min,period,tau);
%figure; plot(etaf)

% Gradient steps 
rho_offset     = etaf;
rho_decomp     = 0.1*etaf;

% Fit set up
fit_offset_dim = [1,0,1,1,1]; fit_offset_dim = fit_offset_dim(1:final_dim);
fit_decomp_dim = [1,1,1,1,1]; fit_decomp_dim = fit_decomp_dim(1:final_dim);

% Gather in Structure
fit_param =struct();
fit_param.Pg = Pg;
fit_param.model = 'poisson';
fit_param.disppct = -1;
fit_param.ite_max = ite_max;
fit_param.rho_offset = rho_offset;
fit_param.rho_decomp = rho_decomp;
fit_param.fit_decomp_dim = fit_decomp_dim;
fit_param.fit_offset_dim = fit_offset_dim;

% Optimizer
opt = 'ADAMNC';
fit_param.beta1 = 0.9;
fit_param.beta2 = 0.999;
fit_param.opt   = opt;



% Collapse Condition/Trial Together if necessary
xdims = size(Xobs);
collapse_tensor  = @(X) reshape(X , [xdims(1:2),prod(xdims(3:end))]);

if collapse_conditions && not(stitch_data); fit_param.observed_data = collapse_tensor(observed_data);
else; fit_param.observed_data = observed_data; end

%% Wrappers & Hyperparameters

% Hyperparams
hyperparams = cell(4,1);



cur = -1
if cur == 1
    hyperparams{1,1} = [5,6];  % tensor rank R
    hyperparams{2,1} = [0]; % lambda_0 group
    hyperparams{3,1} = [0.0];    % alpha_0  MLR
    hyperparams{4,1} = {'MRP'};  %Model str : 'GCP' 'GCP x2' 'MRP'
    
elseif cur == 2
    hyperparams{1,1} = [7,8];  % tensor rank R
    hyperparams{2,1} = [0]; % lambda_0 group
    hyperparams{3,1} = [0.0];    % alpha_0  MLR
    hyperparams{4,1} = {'MRP'};  %Model str : 'GCP' 'GCP x2' 'MRP'
    
elseif cur == 3
    hyperparams{1,1} = [9,10];  % tensor rank R
    hyperparams{2,1} = [0]; % lambda_0 group
    hyperparams{3,1} = [0.0];    % alpha_0  MLR
    hyperparams{4,1} = {'MRP'};  %Model str : 'GCP' 'GCP x2' 'MRP'
    
elseif cur == 4
    hyperparams{1,1} = [5,6];  % tensor rank R
    hyperparams{2,1} = [0.5]; % lambda_0 group
    hyperparams{3,1} = [0.0];    % alpha_0  MLR
    hyperparams{4,1} = {'MRP'};  %Model str : 'GCP' 'GCP x2' 'MRP'
    
elseif cur == 5
    hyperparams{1,1} = [7,8];  % tensor rank R
    hyperparams{2,1} = [0.5]; % lambda_0 group
    hyperparams{3,1} = [0.0];    % alpha_0  MLR
    hyperparams{4,1} = {'MRP'};  %Model str : 'GCP' 'GCP x2' 'MRP'
    
elseif cur == 6
    hyperparams{1,1} = [9,10];  % tensor rank R
    hyperparams{2,1} = [0.5]; % lambda_0 group
    hyperparams{3,1} = [0.0];    % alpha_0  MLR
    hyperparams{4,1} = {'MRP'};  %Model str : 'GCP' 'GCP x2' 'MRP'
    
elseif cur == 7
    hyperparams{1,1} = [7,8];  % tensor rank R
    hyperparams{2,1} = [0.5]; % lambda_0 group
    hyperparams{3,1} = [0.01];    % alpha_0  MLR
    hyperparams{4,1} = {'MRP'};  %Model str : 'GCP' 'GCP x2' 'MRP'
elseif cur == -1
    hyperparams{1,1} = [7,8];  % tensor rank R
    hyperparams{2,1} = [0, 0.5]; % lambda_0 group
    hyperparams{3,1} = [0.0, 0.01];    % alpha_0  MLR
    hyperparams{4,1} = {'MRP'};  %Model str : 'GCP' 'GCP x2' 'MRP'    
end


fit_par = fit_param;

%% Set-up Fit
k_init = 60; %k_init = 36; 

%experimental_parameters.condition_design   = condition_design;
experimental_parameters.collapse_conditions = collapse_conditions;

Nhypers = cellfun(@(Z) size(Z,2), hyperparams)';
models_tot = cell([Nhypers(:)',k_init]);

% Needed for parallel loop
Nypers1 = Nhypers(1);Nypers2 = Nhypers(2); Nypers3 = Nhypers(3); Nypers4 = Nhypers(4);

%% Fit
filename = [resu_folder,'singlefit_',type_data, '_stitching', num2str(stitch_data), '_flipback', num2str(flip_back_data),'_collapsed', num2str(collapse_conditions),'_',cell2mat(kept_region),'_' ,datestr(now,'yyyy_mm_dd_HH_MM')];
disp(filename)

% Train
Xobs_train = Xobs;

% Collapse Condition/Trial Together if necessary
if collapse_conditions
    Xobs_train    = collapse_tensor(Xobs_train);
end

parfor init_cur = 1:k_init % PUT BACK !!
    for hyp_1 = 1:Nypers1
        for hyp_2 = 1:Nypers2
            for hyp_3 = 1:Nypers3
                for hyp_4 = 1:Nypers4
                    disp(['Init: ', num2str(init_cur),'/', num2str(k_init),... 
                        ' HP1: ', num2str(hyp_1), '/', num2str(Nhypers(1)),...
                        ' HP2: ', num2str(hyp_2), '/', num2str(Nhypers(2)),...
                        ' HP3: ', num2str(hyp_3), '/', num2str(Nhypers(3)),...
                        ' model:  ', hyperparams{4}{1,hyp_4}]);
                    
                    % Set current Hyperparameters
                    hyp_cur = cell(1,3);
                    hyp_cur{1,1} = hyperparams{1}(1,hyp_1);
                    hyp_cur{1,2} = hyperparams{2}(1,hyp_2);
                    hyp_cur{1,3} = hyperparams{3}(1,hyp_3);
                    
                    % Current Model
                    mod_cur = hyperparams{4}{1,hyp_4};
                    fit_par = fit_param;
                    
                    % Fit
  
                    results = tensor_wrapper(Xobs_train,mod_cur,fit_par,hyp_cur,sum(observed_data(:)));
                    
                    % Save Model
                    models_tot{hyp_1,hyp_2,hyp_3,hyp_4,init_cur} = results.fit.CP;
                    offset_tot{hyp_1,hyp_2,hyp_3,hyp_4,init_cur} = results.fit.offsets;
                    
                end
            end
        end
    end
end

%%



%%

filename = [resu_folder,'singlefit_',type_data, '_stitching', num2str(stitch_data), '_flipback', num2str(flip_back_data),'_collapsed', num2str(collapse_conditions),'_',cell2mat(kept_region),'_' ,datestr(now,'yyyy_mm_dd_HH_MM')];
save(filename,'models_tot','offset_tot','experimental_parameters','fit_param','hyperparams','-v7.3')

delete(gcp);
exit; 










