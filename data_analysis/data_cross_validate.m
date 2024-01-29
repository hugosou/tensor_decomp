%% Load and organize data

% Add master folders
addpath(genpath('./tensorfact_master'))

data_folder = '/nfs/gatsbystor/hugos/data_sepi_all/';
resu_folder = '/nfs/gatsbystor/hugos/';

%data_folder = '~/Documents/Data/data_sepi_all/';
%resu_folder = '~/Documents/Data/data_sepi_all/';

% Dataset Parameters
flip_back_data = 0;
stitch_data    = 0;

% Which dataset to use: 'mismatch', 'standard'
type_data = 'standard';

% Load
name_data = [data_folder,'data_sepi_',type_data, '_stitching', num2str(stitch_data), '_flipback', num2str(flip_back_data)];
load(name_data)

% Keep regions from : 'N/A','RSPd','RSPg','SC','SUB','V1', 'Hip'
kept_region = {'RSPd','RSPg','SC'};

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
collapse_conditions = 0;

% observed_tensor dim
if collapse_conditions
    final_dim = 3;
else 
    final_dim = ndims(Xobs);
end

experimental_parameters = data_sepi;
experimental_parameters.type_data = type_data;
experimental_parameters.collapse_conditions =collapse_conditions;
experimental_parameters.neuron_group_discriminant=neuron_group_discriminant;

%% Cross-Validation Folders / Leave-Neuron-Out indices
rng(1);

% Sizes of Train/Tests sets, Number of LNO
k_test = 5;                       % Size of the test set
k_max  = 60;                      % Number of cross validation folders
n_test = floor(size(Xobs,1)/2);   % Number of LNO 
N_test = 4;                       % Number of LNO tests;



% Get Test/Train Folders and random neuron index
if stitch_data
    % Test/Train Folders
    [test_folder_outer, train_folder_outer] = xval_idx(1:size(Xobs,4),k_test,k_max);
    
    % If data are stiched together: neurons index are taken randomly
    neurons_lno_idx = zeros(N_test,n_test);
    if N_test==size(Xobs,1) && n_test==1
        neurons_lno_idx = (1:N_test)';
    else
        for nn = 1:N_test
            neurons_lno_idx(nn,:) = randperm(size(Xobs,1),n_test);
        end
    end
    
else
    % Get outer folder ids
    ids = 1:size(Xobs,5);
    [test_folder_outer, train_folder_outer] = xval_idx(ids,k_test,k_max);
    
    % Neurons experiment ids
    observed_data_patern  = squeeze(observed_data(:,1,1,:,1));

    % Number of LNO neurons for each experiment session
    n_test_per_experiment = floor((n_test / size(Xobs,1))*sum(observed_data_patern,1));
    
    % Total (updated) number of LNO neuron
    n_test = sum(n_test_per_experiment);
    
    % LNO ids
    neurons_lno_idx = zeros(N_test,n_test);
    neurons_offset  = [0,cumsum(n_test_per_experiment)];
    for nn = 1:N_test
        for expts=1:size(n_test_per_experiment,2)
            neurons_off  = (1+neurons_offset(expts)):neurons_offset(expts+1);
            neurons_cur  = find(observed_data_patern(:,expts));
            neurons_idx  = randperm(length(neurons_cur),n_test_per_experiment(1,expts));
            neurons_kpt  = neurons_cur(neurons_idx);
            
            % Neurons from expts used for nn-th LNO test
            neurons_lno_idx(nn,neurons_off) = neurons_kpt;
        end
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
collapse_tensor  = @(X,xdims) reshape(X , [xdims(1:2),prod(xdims(3:end))]);


%% Wrappers & Hyperparameters

% Hyperparams
hyperparams = cell(4,1);

cur = 3

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
end


%% Set-up Fit
% Collapse Condition/Trial Together if necessary

experimental_parameters.k_max  = k_max;
experimental_parameters.k_test = k_test;
experimental_parameters.n_test = n_test;
experimental_parameters.N_test = N_test;
experimental_parameters.neurons_lno_idx    = neurons_lno_idx;
experimental_parameters.test_folder_outer  = test_folder_outer;
experimental_parameters.train_folder_outer = train_folder_outer;
experimental_parameters.type_data = type_data;
experimental_parameters.collapse_tensor     = collapse_tensor;
experimental_parameters.collapse_conditions = collapse_conditions;
experimental_parameters.neuron_group_discriminant = neuron_group_discriminant;

Dtot    = cell(1,size(test_folder_outer,1));
Nfolder = size(test_folder_outer,1);
Nhypers = cellfun(@(Z) size(Z,2), hyperparams)';
models_tot = cell([Nhypers(:)',Nfolder]);

% Needed for parallel loop
Nypers1 = Nhypers(1);Nypers2 = Nhypers(2); Nypers3 = Nhypers(3); Nypers4 = Nhypers(4);


%% Fit

filename = [resu_folder,'xval_',type_data, '_stitching', num2str(stitch_data), '_flipback', num2str(flip_back_data),'_collapsed', num2str(collapse_conditions),'_',cell2mat(kept_region),'_' ,datestr(now,'yyyy_mm_dd_HH_MM')];

disp(filename)
parfor folder_id= 1:Nfolder 

    % Folders ID
    cur_test_folders  = test_folder_outer( folder_id,:);
    cur_train_folders = train_folder_outer(folder_id,:);
    
    % Train/Test Sets
    if stitch_data
        Xobs_train = Xobs(:,:,:,cur_train_folders);
        Xobs_test  = Xobs(:,:,:,cur_test_folders);
        observed_data_train = 1;
        observed_data_test  = 1;
    else 
        Xobs_train = Xobs(:,:,:,:,cur_train_folders);
        Xobs_test  = Xobs(:,:,:,:,cur_test_folders);
        observed_data_train = observed_data(:,:,:,:,cur_train_folders);
        observed_data_test  = observed_data(:,:,:,:,cur_test_folders);
    end
      
    % Collapse Condition/Trial Together if necessary
    if collapse_conditions
        Xobs_train = collapse_tensor(Xobs_train,size(Xobs_train));
        Xobs_test  = collapse_tensor(Xobs_test, size(Xobs_test));
        observed_data_train = collapse_tensor(observed_data_train, size(observed_data_train));
        observed_data_test  = collapse_tensor(observed_data_test , size(observed_data_test));
    end
    
    % Init Deviances
    D_folder_test  = zeros([Nhypers(:)', N_test, n_test]);
    D_folder_train = zeros([Nhypers(:)']);
    
    % Loop on hyperparameters
    for hyp_1 = 1:Nypers1
        for hyp_2 = 1:Nypers2
            for hyp_3 = 1:Nypers3
                for hyp_4 = 1:Nypers4
                    disp(['Folder: ', num2str(folder_id),'/', num2str(Nfolder),... 
                        ' HP1: ', num2str(hyp_1), '/', num2str(Nhypers(1)),...
                        ' HP2: ', num2str(hyp_2), '/', num2str(Nhypers(2)),...
                        ' HP3: ', num2str(hyp_3), '/', num2str(Nhypers(3)),...
                        ' model: ', hyperparams{4}{1,hyp_4}]);
                    
                    % Set current Hyperparameters
                    hyp_cur = cell(1,3);
                    hyp_cur{1,1} = hyperparams{1}(1,hyp_1);
                    hyp_cur{1,2} = hyperparams{2}(1,hyp_2);
                    hyp_cur{1,3} = hyperparams{3}(1,hyp_3);
                    
                    % Current Model
                    mod_cur = hyperparams{4}{1,hyp_4};
                    fit_par = fit_param; 
                    fit_par.observed_data = observed_data_train;
                   
                     % Fit
                     results = tensor_wrapper(Xobs_train,mod_cur,fit_par,hyp_cur,sum(observed_data(:)));
                     
                     % Save Model
                     models_tot{hyp_1,hyp_2,hyp_3,hyp_4,folder_id} = results.fit.CP;
                     
                     % Train Deviance
                     D_folder_train(hyp_1,hyp_2,hyp_3,hyp_4) = deviance_poisson(Xobs_train,results.fit.Xhat);
                     
                     % Leave-Neuron_Out (LNO) Analysis
                     for nn = 1:N_test
                         % Trained/Tested Neuron Ids
                         idtest = neurons_lno_idx(nn,:); idtrain = 1:size(Xobs_test,1); idtrain(idtest) =[];
                         
                         % Test Set
                         Xtesti = tensor_remove(Xobs_test,1,idtrain);
                         
                         % Update the observed/missing neurons
                         if not(stitch_data)
                            observed_data_dim = size(observed_data_test);
                            observed_data_tmp = reshape(observed_data_test,[observed_data_dim(1), prod(observed_data_dim(2:end))]);
                            observed_data_lno = reshape(observed_data_tmp(idtrain,:),[length(idtrain), observed_data_dim(2:end)]); 
                            results.fit_param.observed_data = observed_data_lno;
                         end
                         
                         % Predict Xtesti using LNO methods
                         Xpredi = lno_wrapper(results,mod_cur,idtrain,idtest,Xobs_test);
                         
                         % Neuron-Wise Test Deviance
                         D_folder_test(hyp_1,hyp_2,hyp_3,hyp_4, nn,:) = deviance_poisson_nwise(Xtesti, Xpredi);
                        
                     end
                end
            end
        end
    end

    Dtot{1,folder_id}.Dtrain = D_folder_train;
    Dtot{1,folder_id}.Dtest  = D_folder_test;
    
end


filename = [resu_folder,'xval_',type_data, '_stitching', num2str(stitch_data), '_flipback', num2str(flip_back_data),'_collapsed', num2str(collapse_conditions),'_',cell2mat(kept_region),'_' ,datestr(now,'yyyy_mm_dd_HH_MM')];
save(filename,'Dtot','models_tot','experimental_parameters','fit_param','hyperparams','test_folder_outer','train_folder_outer','neurons_lno_idx','-v7.3')

delete(gcp);
exit; 










