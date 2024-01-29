function id_bocks = find_x_blocks(observed_data)
% Format: [[start(b1), end(b1)];
%          [start(b2), end(b2)];
%          [........., .......];
%          [start(bn), end(bn)]];

diff_observed = observed_data;
diff_observed(2:end,:) = diff(observed_data,1);

id_bocks  = find(sum(diff_observed.*(diff_observed>0),2)>0);
id_bocks  = [id_bocks ; size(observed_data,1)+1];
id_bocks  = [id_bocks(1:end-1), id_bocks(2:end)-1];

end