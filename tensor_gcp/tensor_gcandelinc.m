function results = tensor_gcandelinc(Xobs,fit_param,fit_init)
%% TENSOR_GCANDELINC: Generalized CP decomposition of tensor Xobs (with linear constraint)
%                     Xobs ~ f_link([decomp + offsets]) 
%                     decomp = [|CP_1, CP_2, ... CP_N|] OR
%                         OR = [|P_1 x CP_1 x Q_1, P_2 x CP_2 x Q_2, ... P_N x CP_N x Q_N|] (linear constraint)
%                         low rank dynamics 
%                     offset low rank offset
%                     Optimizing Loss = F_link((What+ offset)) - ||Xobs x (What+ offset)||_F^2

% Required Parameters 
if not(isfield(fit_param,'R'));     error('Tensor decomposition order not provided'); end
if not(isfield(fit_param,'model')); error('Noise Model not provided'); end
if not(isfield(fit_param,'opt'));   error('Optimization Method not provided'); end

% Optional Parameters: iteration and convergence threshold
if not(isfield(fit_param,'ite_max')); ite_max = 2000; else; ite_max = fit_param.ite_max;  end
if not(isfield(fit_param,'disppct')); disppct =  5;   else; disppct = fit_param.disppct;  end
if not(isfield(fit_param,'convthr')); convthr = 1e-12;else; convthr = fit_param.convthr;  end
if not(isfield(fit_param,'failcnt')); failcnt = 0;    else; failcnt = fit_param.failcnt;  end
if not(isfield(fit_param,'failmax')); failmax = 10;   else; failmax = fit_param.failmax;  end

% Gradient steps and/or ADAM parameters
if not(isfield(fit_param,'rho_offset')); fit_param.rho_offset = 1e-3*ones(1,ite_max); end
if not(isfield(fit_param,'rho_decomp')); fit_param.rho_decomp = 1e-3*ones(1,ite_max); end
if strcmp(fit_param.opt,'ADAM') || strcmp(fit_param.opt,'ADAMNC')
   if  not(isfield(fit_param,'beta1')); fit_param.beta1 = 0.9; end
   if  not(isfield(fit_param,'beta2')); fit_param.beta2 = 0.999; end
end

% Dimensions to be fitted
if not(isfield(fit_param,'fit_offset_dim')); fit_param.fit_offset_dim = zeros(1,ndims(Xobs)); end
if not(isfield(fit_param,'fit_decomp_dim')); fit_param.fit_decomp_dim =  ones(1,ndims(Xobs)); end
if not(isfield(fit_param,'fit_decomp_lag')); fit_param.fit_decomp_lag =  0;  end
if not(isfield(fit_param,'sprstyc')); fit_param.sprstyc = zeros(1,ndims(Xobs));end

% Get Factor Sizes
[factor_size, fit_param] = get_factor_sizes(Xobs,fit_param);

% Initialize moments (if ADAM), factors and offsets if not provided
if nargin < 3
    factors      = init_factors(factor_size);
    offsets = init_offsets(size(Xobs),fit_param.fit_offset_dim);
    if (strcmp(fit_param.opt,'ADAM') || strcmp(fit_param.opt,'ADAMNC'))
        fit_param.moments = init_moments(factor_size);
    end
    fit_init = struct();  
    fit_init.factors=factors;
    fit_init.offsets=offsets;
else
    
    % Init Factors
    if isfield(fit_init,'factors');factors=fit_init.factors;
    else; factors = init_factors(factor_size); fit_init.factors=factors; end
    
    % Init Offset
    if isfield(fit_init,'offsets');offsets=fit_init.offsets;
    else; offsets = init_offsets(size(Xobs),fit_param.fit_offset_dim); fit_init.offsets=offsets; end
    
    % Init Moments
    if (strcmp(fit_param.opt,'ADAM') || strcmp(fit_param.opt,'ADAMNC'))
        if isfield(fit_init,'moments'); fit_param.moments = fit_init.moments;
        else;fit_param.moments = init_moments(factor_size); end
    end
end

% Get initial tensor decomposition
decomp = tensor_reconstruct_fast(get_constrained_factor(factors, fit_param));

% Get link functions
[fit_param.F_link, fit_param.f_link, fit_param.df_link ] = get_f_links(fit_param.model);


update_offsets = sum(fit_param.fit_offset_dim)>0;
fit_decomp_lag = fit_param.fit_decomp_lag;


% Init and estimate Loss function 
loss_tot = zeros(1,ite_max+1);
loss_tot(1,1) = get_loss(Xobs,get_Xhatinv(decomp,offsets),fit_param.F_link);
if disppct>0
    disp([num2str(0),'/',num2str(ite_max), ' Loss:', num2str(loss_tot(1,1))])
end 


opt_fail = 0;
for ite=1:ite_max
    
    % Update GCP decomposition after an optional lag used to fit the offset
    if (ite>fit_decomp_lag) || not(update_offsets)
        
        if isfield(fit_param,'left_constraint') || isfield(fit_param,'right_constraint')
            [factors,decomp,fit_param] = get_gcandelinc_update(Xobs,decomp,offsets,factors,fit_param,ite);
            
        else 
            [factors,decomp,fit_param] = get_gcp_update(Xobs,decomp,offsets,factors,fit_param,ite);
        end
    end
    
    % Update offset
    if update_offsets
        offsets = get_offset_update(Xobs,decomp,offsets,fit_param,ite);
    end
    
    % Update loss (might be expensive)
    update_loss = not(mod(100 * ite/ite_max,abs(disppct)));
    if update_loss
        loss_tot(1,ite+1) = get_loss(Xobs,get_Xhatinv(decomp,offsets),fit_param.F_link);
    else
        loss_tot(1,ite+1) = loss_tot(1,ite);
    end
    
    % Display and convergence thresholds
    display_loss = mod(100 * ite/ite_max,disppct)==0 && disppct>0;
    if display_loss
        disp([num2str(ite),'/',num2str(ite_max), ' Loss:', num2str(loss_tot(1,ite+1))])
    end
    
    % Break if CV reached or CV failed
    if isnan(loss_tot(1,ite+1)) || abs(loss_tot(1,ite+1))>1e200
        disp('Optimization Failed.'); opt_fail = 1;
        break
    elseif (abs(loss_tot(1,ite+1)-loss_tot(1,ite)) < abs(convthr * loss_tot(1,ite))) && ite>(fit_decomp_lag+10) &&(mod(100 * ite/ite_max,abs(disppct))==0)
        disp(['Converged after ', num2str(ite), ' Itetrations']);loss_tot(1,(ite+1):end)= loss_tot(1,(ite+1));
        break
    end
end


% Save Fit
fit = struct();
fit.Loss_tot = loss_tot;
fit.CP       = factors;
fit.decomp   = decomp;
fit.offsets  = offsets;
fit.Xhat     = fit_param.f_link(get_Xhatinv(decomp,offsets));
if isfield(fit_param,'left_constraint') || isfield(fit_param,'right_constraint')
    fit.left_constraint  = fit_param.left_constraint;
    fit.right_constraint = fit_param.right_constraint;
end

results = struct();
results.fit = fit;
results.fit_param = fit_param;
results.fit_init  = fit_init;


if opt_fail && (failcnt<failmax)    
    fit_param.failcnt    = failcnt+1;     
    fit_param.rho_offset = 0.8* fit_param.rho_offset;
    fit_param.rho_decomp = 0.8* fit_param.rho_decomp;
    disp(['Optimization Failed. Start With Smaller Step Size: ', num2str(failcnt),'/',num2str(failmax)])
    results = tensor_gcandelinc(Xobs,fit_param,fit_init);
end

end


%% Helpers


function factors = init_factors(Adims)
factors = cell(1,length(Adims));

for dimi=1:length(Adims)
    factors{1,dimi} =0.01*randn(Adims(1,dimi),Adims(2,dimi));
end

end

function moments = init_moments(Adims)

% Init Full  CP decomposition
mt  = cell(1,length(Adims));
vt  = cell(1,length(Adims));
gt  = cell(1,length(Adims));

% Initialization
for dimi=1:length(Adims)
    mt{1,dimi} = zeros(Adims(1,dimi),Adims(2,dimi));
    vt{1,dimi} = zeros(Adims(1,dimi),Adims(2,dimi));
    gt{1,dimi} = zeros(Adims(1,dimi),Adims(2,dimi));
end

moments = struct();
moments.mt=mt;
moments.vt=vt;
moments.gt=gt;

end


function [factor_sizes, fit_param] = get_factor_sizes(Xobs,fit_param)
% Get Factor sizes.

% No linear Constraint on Factors
if not(isfield(fit_param,'left_constraint')) && not(isfield(fit_param,'right_constraint'))
    factor_size1 = size(Xobs);
    factor_size2 = fit_param.R*ones(1,ndims(Xobs));
    
else
    % Left Linear Constraint
    if isfield(fit_param,'left_constraint')
        left_constraint_size1 = cellfun(@(PP) size(PP,1),fit_param.left_constraint,'UniformOutput',1);
        left_constraint_size2 = cellfun(@(PP) size(PP,2),fit_param.left_constraint,'UniformOutput',1);
        
        % Check that left constraint is properly dimensioned
        dummy_constraint = not(left_constraint_size1==size(Xobs));
        assert(all(left_constraint_size1(find(dummy_constraint))==1),'Invalid Left Constraint')
        assert(all(left_constraint_size2(find(dummy_constraint))==1),'Invalid Left Constraint')
        
        % Get First dimmensions
        factor_size1 = dummy_constraint.*size(Xobs) + not(dummy_constraint).*left_constraint_size2;

    else
        factor_size1 = size(Xobs);
        left_constraint = num2cell(ones(1,length(factor_size1)));
        fit_param.left_constraint = left_constraint;
    end
    
    % Right Linear Constraint
    if isfield(fit_param,'right_constraint')
        right_constraint_size1 = cellfun(@(PP) size(PP,1),fit_param.right_constraint,'UniformOutput',1);
        right_constraint_size2 = cellfun(@(PP) size(PP,2),fit_param.right_constraint,'UniformOutput',1);
    
        % Check that Right constraint is properly dimensioned
        dummy_constraint = not(right_constraint_size2==fit_param.R*ones(1,ndims(Xobs)));
        assert(all(right_constraint_size1(find(dummy_constraint))==1),'Invalid Right Constraint')
        assert(all(right_constraint_size2(find(dummy_constraint))==1),'Invalid Right Constraint')
        
        % Get Second dimmensions
        factor_size2 = dummy_constraint.*fit_param.R.*ones(1,ndims(Xobs)) + not(dummy_constraint).*right_constraint_size1;
    else
        factor_size2 = fit_param.R*ones(1,ndims(Xobs));
        right_constraint = num2cell(ones(1,length(factor_size2)));
        fit_param.right_constraint = right_constraint;
    end

end

factor_sizes = [factor_size1;factor_size2];

end

function offsets = init_offsets(Xdims, fit_offset_dim)
    offsets_tmp = 0.01*randn(fit_offset_dim.*Xdims+not(fit_offset_dim));
    offsets     = repmat(offsets_tmp, fit_offset_dim + not(fit_offset_dim).*Xdims);
end





function Loss = get_loss(Xobs,Xhatinv,F_link)
    Loss = F_link(Xhatinv) - Xobs(:)'*Xhatinv(:);
end

function Xhatinv = get_Xhatinv(What,Vhat)
    Xhatinv = What + Vhat;
end