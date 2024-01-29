function datam = data_sepi_load(datam,kept_region,neuron_group_discriminant)
%% Build observation tensor from pre-processed data in structure 'datam'
if nargin <3
    neuron_group_discriminant = 'layer';
end

%% Reduce dataset by removing recordings from som brain area if necessary
% Keep regions from : 'N/A','RSPd','RSPg','SC','SUB','V1', 'Hip'
dict_tot = cell(1,3);
dict_tot{1,1} = unique(categorical(datam.record_region));
dict_tot{1,2} = unique(categorical(datam.record_layer));
dict_tot{1,3} = unique(categorical(datam.record_celltype));

record_tot = cell(1,3);
record_tot{1,1} = grp2idx(categorical(datam.record_region));
record_tot{1,2} = grp2idx(categorical(datam.record_layer));
record_tot{1,3} = grp2idx(categorical(datam.record_celltype));


[datam.observed_tensor, dict_tot_out,record_tot_out,kept_neuron] = reduce_dataset(datam.observed_tensor,dict_tot,record_tot, kept_region);

dict_region   = dict_tot_out{1,1};
dict_layer    = dict_tot_out{1,2};
dict_celltype = dict_tot_out{1,3};

record_region   = record_tot_out{1,1};
record_layer    = record_tot_out{1,2};
record_celltype = record_tot_out{1,3};

% Prune observed_data tensor
XDIMS = size(datam.observed_tensor);
ODIMS = size(datam.observed_data);
observed_data = reshape(datam.observed_data, [ODIMS(1), prod(ODIMS(2:end))]);
datam.observed_data = reshape(observed_data(kept_neuron,:),XDIMS);

%% Group neurons based on brain area / cell type / layer
% Neuron Type Penalty: 'ctype','layer','region'
if strcmp(neuron_group_discriminant,'ctype')
    record_ids = [record_celltype,record_layer, record_region];
    dict_tot = cell(1,size(record_ids,2));
    dict_tot{1,end-2} = dict_celltype;
    dict_tot{1,end-1} = dict_layer;
    dict_tot{1,end}   = dict_region;
elseif strcmp(neuron_group_discriminant,'layer')
    record_ids = [record_layer,record_region];
    dict_tot = cell(1,size(record_ids,2));
    dict_tot{1,end-1} = dict_layer;
    dict_tot{1,end}   = dict_region;
elseif strcmp(neuron_group_discriminant,'region')
    record_ids = [record_region];
    dict_tot = cell(1,size(record_ids,2));
    dict_tot{1,end}   = dict_region;
end
 
[indicatorff,neuron_idgrp,neuron_strgp] = get_indicator(record_ids,dict_tot);

% Neuron-type wise constraint
neuron_group = cell(1,length(size(datam.observed_tensor)));
neuron_group{1,1} = indicatorff;

% Simlplify missing data
if all(datam.observed_data(:) == 1)
    datam.observed_data = 1;
end

%% Gather experimental parameters

datam.dict_region   = dict_region;
datam.dict_layr     = dict_layer;
datam.dict_ctype    = dict_celltype;
datam.record_region   = record_region;
datam.record_layer    = record_layer;
datam.record_celltype = record_celltype;
datam.neuron_group  = neuron_group;
datam.neuron_idgrp  = neuron_idgrp;
datam.neuron_strgp  = neuron_strgp;


end









