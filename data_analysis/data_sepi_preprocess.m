%% Load dataset
data_folder = '~/Documents/Data/data_sepi_raw/';
resu_folder = '~/Documents/Data/data_sepi_all/';

% Dataset Parameters
flip_back_data = 1;
stitch_data    = 0;

% Which dataset to use: 'mismatch', 'standard', 'firstsession'
name_data = 'firstsession';

% To save
file_name = [resu_folder,'data_sepi_',name_data, '_stitching', num2str(stitch_data), '_flipback', num2str(flip_back_data)];

% Load
if strcmp(name_data,'standard')
    data_raw     = readtable([data_folder,'SK-dataframe-three-conditions-with-labels.csv']);
elseif strcmp(name_data,'mismatch')
    data_raw    = readtable([data_folder,'SK-dataframe-mismatches-with-labels.csv']);
elseif strcmp(name_data,'firstsession')
    data_raw    = readtable([data_folder,'first-session-experiments-three-conditions.csv']);
end

%% Separate Table

trial_condition = data_raw.TrialCondition;
[trial_direction,directions] = grp2idx(data_raw.InitialDir); % CW = 1, CCW = 2
[id_condition   ,conditions] = grp2idx(trial_condition);
id_experiment = data_raw.Expt_;
id_trial      = data_raw.Trial_;
id_cell       = data_raw.Cell_;
id_bin_number = data_raw.Bin_;
id_region     = data_raw.Region;
id_layer      = data_raw.Layer;
id_celltype   = data_raw.CellType;
position      = data_raw.Position_deg_;
velocity      = data_raw.Speed_deg_s_;
frates        = data_raw.FiringRate; %frates = frates / 10;






%% Split up by trial condition and experiment number

number_experiments = length(unique(id_experiment));   % Total number of experiments
number_conditions  = length(unique(trial_condition)); % Number of experimental conditions

data_split   = cell(number_conditions, number_experiments);

for cond_i = 1:number_conditions
    for exp_j = 1:number_experiments
        
        % Data in condition cond_i and experiment# exp_j
        id_condi_expj = find((id_experiment==exp_j) .* (id_condition == cond_i));
        
        % Split up by trial condition, experiment# and trial#
        data_split_trial = cell(1,length(unique(id_trial(id_condi_expj))));
        
        % Current Number of trial
        number_trials_ij = length(unique(id_trial(id_condi_expj)));
        
        for tria_k =1:number_trials_ij
            
            % Data in condition cond_i, experiment exp_j, trial tria_k
            id_triak = find(id_trial(id_condi_expj)==tria_k);
            id_condi_expj_triak = id_condi_expj(id_triak);
            
            % Current number of recorded number
            Nneuron_cur = length(unique(id_cell(id_condi_expj(id_triak))));
            
            % Current number of time bin
            Nbins_tot   = length(id_triak) / Nneuron_cur;
            
            % Grasp parameters
            id_cell_ijk         = reshape(        id_cell(id_condi_expj_triak),[Nbins_tot, Nneuron_cur]);
            id_bin_number_ijk   = reshape(  id_bin_number(id_condi_expj_triak),[Nbins_tot, Nneuron_cur]);
            id_region_ijk       = reshape(      id_region(id_condi_expj_triak),[Nbins_tot, Nneuron_cur]);
            id_layer_ijk        = reshape(       id_layer(id_condi_expj_triak),[Nbins_tot, Nneuron_cur]);
            id_celltype_ijk     = reshape(    id_celltype(id_condi_expj_triak),[Nbins_tot, Nneuron_cur]);
            position_ijk        = reshape(       position(id_condi_expj_triak),[Nbins_tot, Nneuron_cur]);
            velocity_ijk        = reshape(       velocity(id_condi_expj_triak),[Nbins_tot, Nneuron_cur]);
            frates_ijk          = reshape(         frates(id_condi_expj_triak),[Nbins_tot, Nneuron_cur]);
            trial_direction_ijk = reshape(trial_direction(id_condi_expj_triak),[Nbins_tot, Nneuron_cur]);
            
            % Sanity check.
            assert(Nbins_tot==70, 'Incorrect number of time bins')
            postest = diff(position_ijk');assert(all(postest(:)==0));
            veltest = diff(velocity_ijk');assert(all(veltest(:)==0));
            celtest = diff(id_cell_ijk);  assert(all(celtest(:)==0));
            bintest = diff(id_bin_number_ijk'); assert(all(bintest(:)==0));
            regtest = diff(reshape(grp2idx(id_region(id_condi_expj(id_triak))),  [Nbins_tot, Nneuron_cur])); assert(all(regtest(:)==0));
            lyrtest = diff(reshape(grp2idx(id_layer(id_condi_expj(id_triak))),   [Nbins_tot, Nneuron_cur])); assert(all(lyrtest(:)==0));
            ctytest = diff(reshape(grp2idx(id_celltype(id_condi_expj(id_triak))),[Nbins_tot, Nneuron_cur])); assert(all(ctytest(:)==0));
            
            % Get rid of unecessary data
            bin_cur = id_bin_number_ijk(:,1);
            pos_cur = position_ijk(:,1);
            vel_cur = velocity_ijk(:,1);
            cel_cur = id_cell_ijk(1,:)';
            reg_cur = id_region_ijk(1,:)';
            lyr_cur = id_layer_ijk(1,:)';
            cty_cur = id_celltype_ijk(1,:)';
            dir_cur = trial_direction_ijk(:,1);
            fra_cur = frates_ijk;
            
            % Gather data
            data_split_trial{1,tria_k} = struct();
            data_split_trial{1,tria_k}.id_bin         = bin_cur;
            data_split_trial{1,tria_k}.id_cell        = cel_cur;
            data_split_trial{1,tria_k}.id_region      = reg_cur;
            data_split_trial{1,tria_k}.id_layer       = lyr_cur;
            data_split_trial{1,tria_k}.id_celltype    = cty_cur;
            data_split_trial{1,tria_k}.position       = pos_cur;
            data_split_trial{1,tria_k}.velocity       = vel_cur;
            data_split_trial{1,tria_k}.frates         = fra_cur;
            data_split_trial{1,tria_k}.trial_direction= dir_cur;
            
            % Sanity plot
            %compt = compt+1;
            %subplot(2,1,1); hold on
            %plot(pos_cur+1*compt)
            
            %subplot(2,1,2); hold on
            %plot(vel_cur+1*compt)
            
        end
        
        data_split{cond_i,exp_j} = data_split_trial;
        
    end
end




%% Get total number of trials and recorded neurons
number_cells_tot  = zeros(1,number_experiments);
number_trials_tot = zeros(number_conditions,number_experiments);

for exp_j = 1:number_experiments
    
    % Cell number in exp_j
    number_cell_expj = length(data_split{1,exp_j}{1}.id_cell);
    
    % Check same number of cells in each conditions
    for cond_i = 1:number_conditions
        % Check same number of cells in each conditions and trials
        number_trials_ij = length(data_split{cond_i,exp_j});
        number_trials_tot(cond_i,exp_j) = number_trials_ij;
        for tria_k =1:number_trials_ij
            number_cell_expj_condi_triak = length(data_split{cond_i,exp_j}{tria_k}.id_cell);
            assert(number_cell_expj_condi_triak== number_cell_expj, 'Incorrect Cell Number')
        end
    end
    number_cells_tot(1,exp_j) = number_cell_expj;
end


%% Tensor Dimensions

N_neuron     = sum(number_cells_tot);
T_point      = length(data_split{1}{1}.id_bin);
K_trial      = min(number_trials_tot(:));
L_condition  = number_conditions;
E_experiment = number_experiments;

%% Build tensors
neuron_id_final = cumsum([1,number_cells_tot]);

% Gather observed data
if stitch_data
    % Do NOT take into account multiple mice
    observed_tensor = zeros(N_neuron,T_point,L_condition,K_trial);
    observed_data   = zeros(N_neuron,T_point,L_condition,K_trial);
    for cond_i = 1:L_condition
        for exp_j = 1:E_experiment
            for tria_k =1:K_trial
                % Currently recorded neurons
                neuron_id_cur = neuron_id_final(exp_j):neuron_id_final(exp_j+1)-1;
                
                % Observed/Unobersed Spikes
                observed_tensor(neuron_id_cur,:,cond_i,tria_k) = data_split{cond_i,exp_j}{1,tria_k}.frates';
                observed_data(  neuron_id_cur,:,cond_i,tria_k) = 1;
            end
        end
    end
    
else
    % Take into account multiple mice
    observed_tensor = zeros(N_neuron,T_point,L_condition,K_trial);
    observed_data   = zeros(N_neuron,T_point,L_condition,K_trial);
    for cond_i = 1:L_condition
        for exp_j = 1:E_experiment
            for tria_k =1:K_trial
                % Currently recorded neurons
                neuron_id_cur = neuron_id_final(exp_j):neuron_id_final(exp_j+1)-1;
                
                % Observed/Unobersed Spikes
                observed_tensor(neuron_id_cur,:,cond_i,exp_j,tria_k) = data_split{cond_i,exp_j}{1,tria_k}.frates';
                observed_data(  neuron_id_cur,:,cond_i,exp_j,tria_k) = 1;
            end
        end
    end
    
end

% From frates to spike counts
observed_tensor = observed_tensor/10;

% Gather neuron Region/Layer/Type
record_region   = cell(N_neuron,1);
record_layer    = cell(N_neuron,1);
record_celltype = cell(N_neuron,1);

for exp_j = 1:E_experiment
    % Currently recorded neurons
    neuron_id_cur = neuron_id_final(exp_j):neuron_id_final(exp_j+1)-1;
    
    record_region(neuron_id_cur,1)   = data_split{1,exp_j}{1}.id_region;
    record_layer(neuron_id_cur,1)    = data_split{1,exp_j}{1}.id_layer;
    record_celltype(neuron_id_cur,1) = data_split{1,exp_j}{1}.id_celltype;
end

% Stimulus variables
position = data_split{1}{1}.position;
velocity = data_split{1}{1}.velocity;

% Initial direction
direction = zeros(L_condition,E_experiment,K_trial);
for cond_i = 1:L_condition
    for exp_j = 1:E_experiment
        for tria_k =1:K_trial
            direction(cond_i,exp_j,tria_k) = data_split{cond_i,exp_j}{1,tria_k}.trial_direction(1);
        end
    end
end

% CCW: 1 CW: -1
direction = (direction *2 - 3);

% Time Flip Data
if flip_back_data
    for cond_i = 1:L_condition
        for exp_j = 1:E_experiment
            for tria_k =1:K_trial
                % Clock-wise Rotation
                if (direction(cond_i,exp_j,tria_k) == -1)
                    if stitch_data
                        observed_tensor(:,:,cond_i,tria_k) ...
                            = observed_tensor(:,T_point:-1:1,cond_i,tria_k);
                    else
                        observed_tensor(:,:,cond_i,exp_j,tria_k) ...
                            = observed_tensor(:,T_point:-1:1,cond_i,exp_j,tria_k);
                    end
                end
            end
        end
    end
end

%% Gather All
data_sepi = struct();

% Spikes
data_sepi.observed_tensor = observed_tensor;
data_sepi.observed_data   = observed_data;

% Stimulus
data_sepi.direction = direction;
data_sepi.position  = position;
data_sepi.velocity  = velocity;
data_sepi.condition = conditions;

% Neuron Groups
data_sepi.record_region   = record_region;
data_sepi.record_layer    = record_layer;
data_sepi.record_celltype = record_celltype;


%% Save
save(file_name, 'data_sepi')
