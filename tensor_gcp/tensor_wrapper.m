function [results,results_tmp] = tensor_wrapper(X,method,fit_param,fit_hyperparams, num_data)

if nargin <5 
   num_data = numel(X);
elseif num_data == 1
   num_data = numel(X); 
end

fit_param.R       = fit_hyperparams{1,1};
fit_param.lambdag = fit_hyperparams{1,2}.* sqrt(num_data).*ones(size(fit_param.Pg{1,1},2),fit_param.R ); 


if strcmp(method,'GCP')
    fit_param.alphas = zeros(1,ndims(X));
    results = tensor_mrp(X,fit_param);

elseif strcmp(method,'GCPx2')
    fit_param.alphas = zeros(1,ndims(X));
    
    %1st Pass
    results_tmp = tensor_mrp(X,fit_param);
    
    % Init 2nd Pass
    fit_init = struct();
    fit_init.factors  = results_tmp.fit.CP;
    fit_init.offsets  = results_tmp.fit.offsets;
    fit_init.moments  = results_tmp.fit_param.moments ;
    
    % 2nd Pass
    results = tensor_mrp(X,fit_param,fit_init);
    
elseif strcmp(method,'MRP')
    fit_param.alphas = zeros(1,ndims(X));
    
    %1st Pass
    results_tmp = tensor_mrp(X,fit_param);
    
    % Init 2nd Pass
    fit_init = struct();
    fit_init.factors  = results_tmp.fit.CP;
    fit_init.offsets  = results_tmp.fit.offsets;
    fit_init.moments  = results_tmp.fit_param.moments ;
    
    % Get factors MLR threshold from factors SVD
    alpha0 = fit_hyperparams{1,3};
    alphai = cell2mat(cellfun(@(x) max(svd(x,'econ')),fit_init.factors , 'UniformOutput',false));
    
    
    % Be careful about this !
    fit_param.alphas  = alpha0.*ones(1,ndims(X)).* sqrt(numel(X)).*alphai;

    %fit_param.alphas  = alpha0.*[1,1,1].* sqrt(numel(X)).*alphai;
    
    % 2nd PAss
    results = tensor_mrp(X,fit_param,fit_init);
    
end


end











