function [observed_tensor, experimental_parameters] = data_sepi_load(datam,kept_region,neuron_group_discriminant)
%% Build observation tensor from pre-processed data in structure 'datam'
%% Load recordings and build observation tensor

if nargin <3
    neuron_group_discriminant = 'layer';
end

% Experimental Condition ID
trial_cond = datam.trial_cond;
dict_cond  = unique(trial_cond);

% Unfold trials and Separate XP conditions
data_sepim_unfolded = unfold_bin(datam);

% Convert The recorded Region Into an int array
RegionDict =  unique(data_sepim_unfolded.record_region);
RegionId   = grp2idx(data_sepim_unfolded.record_region);
data_sepim_unfolded.record_region = RegionId;
data_sepim_unfolded.record_region_dict = RegionDict;

% Convert The recorded Region Into an int array
LayerDict =  unique(data_sepim_unfolded.record_layr);
LayerId   = grp2idx(data_sepim_unfolded.record_layr);
data_sepim_unfolded.record_layr = LayerId;
data_sepim_unfolded.record_layr_dict = LayerDict;

% Convert The cell TypeInto an int array
CellTDict =  unique(data_sepim_unfolded.record_ctype);
CellTId   = grp2idx(data_sepim_unfolded.record_ctype);
data_sepim_unfolded.record_ctype = CellTId;
data_sepim_unfolded.record_ctype_dict = CellTDict;

% Grasp All Conditions
L = length(dict_cond);

% Grasp data condition wise
frates_all  = cell(1,L);
ktrials_all = zeros(1,L);

for ll = 1:L
    data_ll =  get_cond_data(data_sepim_unfolded, dict_cond(ll));
    frates_all{1,ll}  = permute(reshape_data_bin_trial(data_ll,'fir_rate'), [2,1,3]);
    ktrials_all(1,ll) = size(frates_all{1,ll},3);
end

% Grasp XP variable of interest
posit_temp = reshape_data_bins(data_ll,'posit_deg'); posit_eg = posit_temp(:,1);
speed_temp = reshape_data_bins(data_ll,'speed_deg'); speed_eg = speed_temp(:,1);

% Neuron Types
record_region = permute(reshape_data_bin_trial(data_ll,'record_region'), [2,1,3]);
record_layr   = permute(reshape_data_bin_trial(data_ll,'record_layr'), [2,1,3]);
record_ctype  = permute(reshape_data_bin_trial(data_ll,'record_ctype'), [2,1,3]);
dict_region = data_sepim_unfolded.record_region_dict;
dict_layr   = data_sepim_unfolded.record_layr_dict;
dict_ctype  = data_sepim_unfolded.record_ctype_dict;

% Limit the analysis to same # of XP in all conditions
K= min(ktrials_all);
for ll = 1:L
    frates_all{1,ll}  = frates_all{1,ll}(:,:,1:K);
end

record_region = record_region(:,:,1:K);
record_layr   = record_layr(:,:,1:K);
record_ctype  = record_ctype(:,:,1:K);

% Sanity Check 1 + Get Recording Region and Layer for each Independent cell
assert(all(all(all(record_region-record_region(:,1,1) == 0)))); record_region = record_region(:,1,1);
assert(all(all(all(record_layr-record_layr(:,1,1) == 0))))    ; record_layr   = record_layr(:,1,1);
assert(all(all(all(record_ctype-record_ctype(:,1,1) == 0))))  ; record_ctype = record_ctype(:,1,1);


% Sanity Check 2 : Make sure the dataset is well dimensioned
assert(all(all(diff(cell2mat(cellfun(@(x) size(x)',frates_all , 'UniformOutput',false))',1)==0)))

% Independent Neurons
N = size(frates_all{1,1},1);

% Time Points
T = size(frates_all{1,1},2);

% Build Obervation tensor
observed_tensor = zeros(N,T,L,K);
for ll = 1:L
    observed_tensor(:,:,ll,:) = frates_all{1,ll};
end

% Convert Firing rate to spike count
observed_tensor = observed_tensor/10;

%% Reduce dataset by removing recordings from som brain area if necessary
% Keep regions from : 'N/A','RSPd','RSPg','SC','SUB','V1', 'Hip'
dict_tot = cell(1,3);
dict_tot{1,1} = dict_region;
dict_tot{1,2} = dict_layr;
dict_tot{1,3} = dict_ctype;

record_tot = cell(1,3);
record_tot{1,1} = record_region;
record_tot{1,2} = record_layr;
record_tot{1,3} = record_ctype;

[observed_tensor, dict_tot_out,record_tot_out] = reduce_dataset(observed_tensor,dict_tot,record_tot, kept_region);

dict_region = dict_tot_out{1,1};
dict_layr   = dict_tot_out{1,2};
dict_ctype  = dict_tot_out{1,3};

record_region = record_tot_out{1,1};
record_layr   = record_tot_out{1,2};
record_ctype  = record_tot_out{1,3};

%% Group neurons based on brain area / cell type / layer
% Neuron Type Penalty: 'ctype','layer','region'
if strcmp(neuron_group_discriminant,'ctype')
    record_ids = [record_ctype,record_layr, record_region];
    dict_tot = cell(1,size(record_ids,2));
    dict_tot{1,end-2} = dict_ctype;
    dict_tot{1,end-1} = dict_layr;
    dict_tot{1,end}   = dict_region;
elseif strcmp(neuron_group_discriminant,'layer')
    record_ids = [record_layr,record_region];
    dict_tot = cell(1,size(record_ids,2));
    dict_tot{1,end-1} = dict_layr;
    dict_tot{1,end}   = dict_region;
elseif strcmp(neuron_group_discriminant,'region')
    record_ids = [record_region];
    dict_tot = cell(1,size(record_ids,2));
    dict_tot{1,end}   = dict_region;
end
 

[indicatorff,neuron_idgrp,neuron_strgp] = get_indicator(record_ids,dict_tot);

% Neuron-type wise constraint
neuron_group = cell(1,length(size(observed_tensor)));
neuron_group{1,1} = indicatorff;

%% Gather experimental parameters
experimental_parameters = struct();
experimental_parameters.dict_cond     = dict_cond;
experimental_parameters.dict_region   = dict_region;
experimental_parameters.dict_layr     = dict_layr;
experimental_parameters.dict_ctype    = dict_ctype;
experimental_parameters.record_region = record_region;
experimental_parameters.record_layr   = record_layr;
experimental_parameters.record_ctype  = record_ctype;
experimental_parameters.neuron_group  = neuron_group;
experimental_parameters.neuron_idgrp  = neuron_idgrp;
experimental_parameters.neuron_strgp  = neuron_strgp;
experimental_parameters.position      = posit_eg;
experimental_parameters.velocity      = speed_eg;


end



%% Helper Functions
function conditionned_struct = get_cond_data(data, condition)

conditionned_struct = struct();
trial_cond=data.trial_cond;
exp_cond_id = find(trial_cond == condition); % Vestibular;
conditionned_struct.id_tot        = data.id_tot(exp_cond_id);
conditionned_struct.exp_id        = data.exp_id(exp_cond_id);
conditionned_struct.cell_id       = data.cell_id(exp_cond_id);
conditionned_struct.bin_id        = data.bin_id(exp_cond_id);
conditionned_struct.trial_id      = data.trial_id(exp_cond_id);
conditionned_struct.posit_deg     = data.posit_deg(exp_cond_id);
conditionned_struct.speed_deg     = data.speed_deg(exp_cond_id);
conditionned_struct.fir_rate      = data.fir_rate(exp_cond_id);
conditionned_struct.trial_cond    = data.trial_cond(exp_cond_id);
conditionned_struct.record_region = data.record_region(exp_cond_id);
conditionned_struct.record_layr   = data.record_layr(exp_cond_id);
conditionned_struct.record_ctype  = data.record_ctype(exp_cond_id);

end

