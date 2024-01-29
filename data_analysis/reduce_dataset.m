function [Xout, dict_tot_out,record_tot_out,kept_neuron] = reduce_dataset(Xin,dict_tot,record_tot, kept_var,ref)
% Discard some cell type/region/layer

dict_tot_out   = cell(1,length(dict_tot));
record_tot_out = cell(1,length(record_tot));

if nargin<5
    ref=1;
end

nonref = 1:length(dict_tot);
nonref(ref) = [];
dict_ref = dict_tot{1,ref};
record_ref  = record_tot{1,ref};

kept_id_tot = cell(1,length(dict_tot));


kept_ref_id = zeros(1,length(kept_var));
for kid = 1:length(kept_var)

    rid = find(dict_ref==kept_var{1,kid});
    if not(isempty(rid))
        kept_ref_id(1,kid) = rid;
    end
end

kept_neuron = find(sum(record_ref==kept_ref_id,2));
kept_id_tot{1,ref} = kept_ref_id;
for nonref_i = nonref
    kept_id_tot{1,nonref_i}  = unique(record_tot{1,nonref_i}(kept_neuron))';
end


remapping_tot =cell(1,length(dict_tot));

for ii=1:length(dict_tot)
    dict_cur = dict_tot{1,ii};
    record_cur  = record_tot{1,ii};
    remapping_tot{1,ii} = zeros(1,length(dict_cur));
    kept_id_cur = kept_id_tot{1,ii};
    for jj=1:length(kept_id_cur)
        remapping_tot{1,ii}(1,kept_id_cur(1,jj)) = jj;
    end
    
    record_tot_out{1,ii} = remapping_tot{1,ii}(record_cur(kept_neuron))';
    dict_tot_out{1,ii}   = dict_cur(kept_id_cur);
end

dim_tmp = size(Xin);
Xin_1  = tensor_unfold(Xin,1);
Xout_1 = Xin_1(kept_neuron,:);
Xout   = tensor_fold(Xout_1,[length(kept_neuron),dim_tmp(2:end)],1);


end
