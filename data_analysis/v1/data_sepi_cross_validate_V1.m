%% Load and organize data

% Add master folders
addpath(genpath('./tensorfact_master'))

% Machine Path
folder = '/nfs/ghome/live/hugos/Documents/project_tensors/';
folder = '/home/sou/Documents/Data/';
data_folder = [folder, 'data_sepi/'];
resu_folder = [folder, 'dresults/'];


%data_folder = '/nfs/gatsbystor/hugos/';
%resu_folder = '/nfs/gatsbystor/hugos/';

% Choose dataset: 'standard', 'mismatch' 
dataset_str = 'standard';

% Load PRE-PROCESSED data
if strcmp(dataset_str, 'standard');load([data_folder,'data_sepim']);
elseif strcmp(dataset_str, 'mismatch') ;load([data_folder,'data_sepim_mismatch']);
end

% Keep regions from : 'N/A','RSPd','RSPg','SC','SUB','V1', 'Hip'
%kept_region = {'RSPd','RSPg','SC'};
kept_region = {'RSPd','RSPg','SC','SUB','V1'};

% Use a 3D or 4D tensor
collapse_conditions = 1;

% Attribute to group neurons: 'ctype','layer','region'
neuron_group_discriminant = 'layer';

% Build Tensor
[Xobs, experimental_parameters] = data_sepi_load(data_sepim,kept_region,neuron_group_discriminant);

% Extract experimental parameters
param_names = fieldnames(experimental_parameters);
for parami = 1:length(param_names)
    eval([param_names{parami} '=experimental_parameters.' param_names{parami},';']);
end



%% Cross-Validation Parameters

% Sizes for outer and inner folder of nested cross validation (LOO if = 1)
k_test = 5;                       % Size of the test set
k_max  = 60;                      % Number of cross validation folders
n_test = floor(size(Xobs,1)/2);   % Number of Leave Neuron Out 
N_test = 4;                       % Number of LNO tests;

% Get outer folder ids
ids = 1:size(Xobs,4);
[test_folder_outer, train_folder_outer] = xval_idx(ids,k_test,k_max);

% Permutations for LNO analysis
perm_tot = zeros(N_test,n_test);
if N_test==size(Xobs,1) && n_test==1
    perm_tot = (1:N_test)';
else
    for nn = 1:N_test
        perm_tot(nn,:) = randperm(size(Xobs,1),n_test);
    end
end


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
fit_offset_dim = [1,0,1,0];

% Gather in Structure
fit_param =struct();
fit_param.Pg = Pg;
fit_param.model = 'poisson';
fit_param.disppct = -1;
fit_param.ite_max = ite_max;
fit_param.rho_offset = rho_offset;
fit_param.rho_decomp = rho_decomp;
fit_param.fit_decomp_dim = ones(1,ndims(Xobs)-collapse_conditions);
fit_param.fit_offset_dim = fit_offset_dim(1:(ndims(Xobs)-collapse_conditions));

% Optimizer
opt = 'ADAMNC';
fit_param.beta1 = 0.9;
fit_param.beta2 = 0.999;
fit_param.opt   = opt;

%% Wrappers & Hyperparameters

% Hyperparams
model_str     = {'GCP','GCPx2','MRP'};
fit_param_tot = cell(1,size(model_str,2));
hyperparams   = cell(3,size(model_str,2));

for model_id =1:size(model_str,2)
    fit_param_tot{1,model_id} = fit_param;
    
    hyperparams{1,model_id} = [3,5,7,8,10,14,25];% tensor rank R 
    hyperparams{2,model_id} = [0,0.5];       % lambda_0 group
    hyperparams{3,model_id} = [0.0,0.005];   % alpha_0  MLR

end
    

%% Set-up Fit
% Collapse Condition/Trial Together if necessary
collapse_tensor  = @(X) reshape(permute(X,[1,2,4,3]), [size(X,1),size(X,2),size(X,3)*size(X,4)]);
condition_design = repelem(eye(size(Xobs,3)), 1,k_test);
experimental_parameters.condition_design = condition_design;
experimental_parameters.collapse_conditions=collapse_conditions;

Nfolder = size(test_folder_outer,1);
Nhypers = [length(hyperparams{1,1}),length(hyperparams{2,1}),length(hyperparams{3,1})];
Nmodels = size(model_str,2);

Dtot       = cell(1,size(test_folder_outer,1));
models_tot = cell([Nmodels,Nhypers(:)',Nfolder]);

% Needed for parallel loop
Nypers1 = Nhypers(1);Nypers2 = Nhypers(2);Nypers3 = Nhypers(3);

%% Fit

filename = [resu_folder,'xval_',dataset_str,'_',cell2mat(kept_region),'_collapsed', num2str(collapse_conditions),'_' ,datestr(now,'yyyy_mm_dd_HH_MM')];
disp(filename)
parfor folder_id= 1:Nfolder
    
    % Folders ID
    cur_test_folders  = test_folder_outer(folder_id,:);
    cur_train_folders = train_folder_outer(folder_id,:);
    
    % Train/Test Sets
    Xobs_train = Xobs(:,:,:,cur_train_folders);
    Xobs_test  = Xobs(:,:,:,cur_test_folders);
      
    % Collapse Condition/Trial Together if necessary
    if collapse_conditions
        Xobs_train = collapse_tensor(Xobs_train);
        Xobs_test  = collapse_tensor(Xobs_test);
    end
    
    % Init Deviances
    D_folder_test  = zeros([Nmodels,Nhypers(:)', N_test, n_test]);
    D_folder_train = zeros([Nmodels,Nhypers(:)']);
    
    for hyp_1 = 1:Nypers1
        for hyp_2 = 1:Nypers2
            for hyp_3 = 1:Nypers3
                for model_id = 1:Nmodels
                    disp(['Folder: ', num2str(folder_id), '/', num2str(size(test_folder_outer,1)),...
                          ' HP1: ', num2str(hyp_1), '/', num2str(Nhypers(1)),...
                          ' HP2: ', num2str(hyp_2), '/', num2str(Nhypers(2)),...
                          ' HP3: ', num2str(hyp_3), '/', num2str(Nhypers(3)),...
                          ' model:  ', model_str{1,model_id}]);
                     
                     % Set current Hyperparameters
                     hyp_cur = cell(1,3);
                     hyp_cur{1,1} = hyperparams{1,model_id}(1,hyp_1);
                     hyp_cur{1,2} = hyperparams{2,model_id}(1,hyp_2);
                     hyp_cur{1,3} = hyperparams{3,model_id}(1,hyp_3);
                     
                     % Current Model
                     mod_cur = model_str{1,model_id};
                     fit_par = fit_param_tot{1,model_id};
                     
                     % Fit
                     results = tensor_wrapper(Xobs_train,mod_cur,fit_par,hyp_cur);    
                          
                     % Save Model
                     models_tot{model_id,hyp_1,hyp_2,hyp_3,folder_id} = results.fit.CP;
                     
                     % Train Deviance
                     D_folder_train(model_id,hyp_1,hyp_2,hyp_3) = deviance_poisson(Xobs_train,results.fit.Xhat);
                     
                     % Leave-Neuron_Out (LNO) Analysis
                     for nn = 1:N_test
                         % Trained/Tested Neuron Ids
                         idtest = perm_tot(nn,:); idtrain = 1:size(Xobs_test,1); idtrain(idtest) =[];
                         
                         % Test Set
                         Xtesti = tensor_remove(Xobs_test,1,idtrain);
                         
                         % Predict Xtesti using LNO methods
                         Xpredi = lno_wrapper(results,mod_cur,idtrain,idtest,Xobs_test);
                         
                         % Neuron-Wise Test Deviance
                         D_folder_test(model_id,hyp_1,hyp_2,hyp_3, nn,:) = deviance_poisson_nwise(Xtesti, Xpredi);
                         
                     end
                end
            end
        end
    end

    Dtot{1,folder_id}.Dtrain = D_folder_train;
    Dtot{1,folder_id}.Dtest  = D_folder_test;
    
end

filename = [resu_folder,'xval_',dataset_str,'_',cell2mat(kept_region),'_collapsed', num2str(collapse_conditions),'_' ,datestr(now,'yyyy_mm_dd_HH_MM')];
save(filename,'Dtot','models_tot','experimental_parameters','fit_param_tot','hyperparams','test_folder_outer','train_folder_outer','perm_tot','-v7.3')

delete(gcp);
exit; 










