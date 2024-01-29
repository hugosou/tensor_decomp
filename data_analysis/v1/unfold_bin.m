function data_unfolded = unfold_bin(data,doPlot)
% The original dataset separated forward-backward spin

if nargin<2
    doPlot=0;
end

data_unfolded = data;

bin_id = data.bin_id;
id_tot = data.id_tot;

id_bin_start = find(abs(diff(bin_id)) == max(abs(diff(bin_id))))+1;
id_bin_start = [1;id_bin_start];
id_bin_start_pruned_forw = id_bin_start(1:2:end);
id_bin_start_pruned_back = id_bin_start(2:2:end);


assert(length(id_bin_start_pruned_forw) == length(id_bin_start_pruned_back))

is_forw = id_tot*0;
for ii=1:length(id_bin_start_pruned_back)
    is_forw(id_bin_start_pruned_forw(ii):id_bin_start_pruned_back(ii)-1) = 1;
end
is_back = 1-is_forw;

bin_unfolded = bin_id + max(bin_id)*is_back;

data_unfolded.bin_id=bin_unfolded;
data_unfolded.bin_id_folded = bin_id;

if doPlot
    figure;hold on
    scatter(id_tot, bin_id, 'filled')
    scatter(id_tot, bin_unfolded)
    plot(id_tot,is_back)
    xlim([1,5000])
end

end
