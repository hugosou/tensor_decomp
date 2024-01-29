
function y = reshape_data_bins(data,variable)
    bin_id = data.bin_id;
    unique_bin = unique(bin_id);
    
    x = data.(variable);
    y = zeros(length(unique_bin),length(bin_id)/(length(unique_bin)));
    
    Tbin = sum(bin_id==unique_bin(1));
    for ii=1:length(unique_bin)
        curbin = find(bin_id==unique_bin(ii));   
        assert(Tbin == length(curbin));
        y(unique_bin(ii),:) = x(curbin);  
        

    end
   
end