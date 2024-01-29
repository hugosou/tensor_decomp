function y =reshape_data_bin_trial(data,variable)


bin_reshaped_trials = reshape_data_bins(data,'trial_id');
bin_reshaped_trials_tmp = bin_reshaped_trials(1,:);
bin_reshaped_VOI = reshape_data_bins(data, variable);


max_n_trial = max(bin_reshaped_trials_tmp);
cou_n_trial = zeros(1,max_n_trial);
for ii=1:max_n_trial
    cou_n_trial(ii) = sum(bin_reshaped_trials_tmp==ii);
end

n_trial_tmp = find(cou_n_trial==max(cou_n_trial));
n_trial = n_trial_tmp(end);


n_l     = length(find(bin_reshaped_trials_tmp==1));

y = zeros(size(bin_reshaped_trials,1),n_l, n_trial);


for ii=1:n_trial
   y_n_trial_n_l =  bin_reshaped_VOI(:, find(bin_reshaped_trials_tmp==ii));
   assert(size(y_n_trial_n_l,2)==n_l)
   y(:,:,ii) =  y_n_trial_n_l;
end

end

