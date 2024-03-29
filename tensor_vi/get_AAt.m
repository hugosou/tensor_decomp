function [AAt,MMt] = get_AAt(CP_mean,CP_variance)
% INPUTS : CP decomposition mean    : {E(A1), ..., E(AD)}
%          CP decomposition variance: {V(A1), ..., V(AD)}
%          E(An)(i,:) = m_in          ~ 1 x R
%          V(An)(i,:) = Vec(Sigma_in) ~ 1 x RR
%
% OUPUTS : CP decomposition: AAt = {AA1, ..., AAD}
%          AAn(i,:) = Vec(m_in'm_in + Sigma_in)  ~ 1 x RR
% TODO: Make it more efficient with Khatri-Rao product ?

% Tensor Rank
R = size(CP_mean{1},2);

% Get mm' for each mean vector of each factor
%MMt = cellfun(@(Z) cell2mat(...
%     cellfun(@(Y) reshape(Y(:)*Y(:)',1,R*R) , num2cell(Z,2) , 'UniformOutput', false)...
%     ), CP_mean, 'UniformOutput', false);

% Take the outter product of each rows
MMt = cellfun(@(X) pagemtimes(permute(X, [2,3,1]) , permute(X, [3,2,1])),...
    CP_mean, 'UniformOutput', false);

% Reshape each rows
MMt = cellfun(@(Z) reshape(Z, [size(Z,1)*size(Z,2),size(Z,3)])', ...
    MMt, 'UniformOutput', false);
 
% <AA'> = mm' + Sigma_in
AAt = cellfun(@(X,Y) X + Y,MMt, CP_variance, 'UniformOutput', false); 

end

