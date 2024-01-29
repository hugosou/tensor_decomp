function MTTKRP =  mttkrp_sparse(observed_compact, observed_indices, factors,dimn)
% Sparse Matrix tensor Khatri-Rao Product
% Adapted from Brett W. Bader, Tamara G. Kolda and others

% Dimensions of the problem
sizeX = cellfun(@(Z) size(Z,1), factors);
N = size(observed_indices,2);
R = size(factors{1},2);

% Dimension over which to take the Khatri Rao product
not_dimn = setdiff(1:N,dimn);
MTTKRP = zeros(sizeX(dimn),R);

for r = 1:R
    
    % Grasp r-th column of each factor
    factor_r = cell(N,1);
    for i = not_dimn
        factor_r{i} = factors{i}(:,r);
    end
    
    % Init product
    observed_compact_new = observed_compact;
    
    % Sparse Khatri-Rao product for each dimension
    for ndimn = not_dimn
        idx = observed_indices(:,ndimn);
        factor_rn = factor_r{ndimn};         % extract nth vector
        factor_rn_extended = factor_rn(idx); % stretch out the vector
        
        observed_compact_new = observed_compact_new .* factor_rn_extended;
    end
    
    
    % Gather all
    observed_indices_new = observed_indices(:, dimn);
    newsiz = sizeX(dimn);
    
    MTTKRP(:,r) = accumarray(observed_indices_new,observed_compact_new, [newsiz,1 ]);

end

end