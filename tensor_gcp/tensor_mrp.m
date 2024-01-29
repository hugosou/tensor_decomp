function results = tensor_mrp(Xobs,fit_param,fit_init)
%% TENSOR_MRP: Multi-Rank (generalized) Polyadic decomposition of tensor Xobs (with linear constraint)
%   Xobs   =  f_link(decomp + offsets) 
%   offset =  low rank offset
%   decomp =  [|CP_1, CP_2, ... CP_N|] where  CP_i is the i-th Canonical Polyadic Factor
%
%   optional: Left and Right constraint CP_i := L_i x factor_i x R_i
%   optional: Nuclear norm penalty ||factor_i||*
%             Using Alternate Direction of Multipliers: auxiliary and lagrange multipliers
%              
%   Hugo.S                    

% Set/Get default parameters 

if nargin <3
   fit_init = struct(); 
end

[fit_param, fit_init] = init_all(Xobs, fit_param, fit_init);

% Get initial decomposition
factors   = fit_init.factors;
offsets   = fit_init.offsets;
auxiliary = fit_init.auxiliary;
decomp    = tensor_reconstruct_fast(get_constrained_factor(factors, fit_param));

% Some Fit Params
ite_max = fit_param.ite_max;
disppct = fit_param.disppct;
convthr = fit_param.convthr;
failcnt = fit_param.failcnt;
failmax = fit_param.failmax; 

mlr_penalty     = sum(abs(fit_param.alphas))> 0;
update_offsets  = sum(fit_param.fit_offset_dim)>0;
fit_decomp_lag  = fit_param.fit_decomp_lag;

% Init and estimate Loss function 
loss_tot = zeros(1,ite_max+1);
loss_tot(1,1) = get_loss(Xobs,get_Xhatinv(decomp,offsets),fit_param.F_link,fit_param.observed_data);
if disppct>0
    disp([num2str(0),'/',num2str(ite_max), ' Loss:', num2str(loss_tot(1,1))])
end 

opt_fail = 0;
for ite=1:ite_max
    
    % Update decomp
    if (ite>fit_decomp_lag) || not(update_offsets)
        [factors, decomp, auxiliary, fit_param,Nhats,ZsNNs] = get_mrp_update(Xobs,decomp,offsets,factors,auxiliary,fit_param,ite);   
    end
    
    % Update offset
    if update_offsets
        offsets = get_offset_update(Xobs,decomp,offsets,fit_param,ite);
    end
    
    % Update loss 
    update_loss = not(mod(100 * ite/ite_max,abs(disppct)));
    if update_loss
        
        % -logL for ExpFam
        loss_tmp = get_loss(Xobs,get_Xhatinv(decomp,offsets),fit_param.F_link,fit_param.observed_data); 
        
        % Rank Penalty
        if mlr_penalty
            loss_tmp = loss_tmp + get_mlr_penalty(auxiliary, factors, fit_param, ZsNNs);
        end
        
        loss_tot(1,ite+1) = loss_tmp ;
    else
        loss_tot(1,ite+1) = loss_tot(1,ite);
    end
    
    % Display Loss 
    display_loss = mod(100 * ite/ite_max,disppct)==0 && disppct>0;
    if display_loss
        disp([num2str(ite),'/',num2str(ite_max), ' Loss:', num2str(loss_tot(1,ite+1),10), ' Nhat: [' , num2str(Nhats),']' ])
    end
    
    % Break if fit converged or failed
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
fit.offsets   = offsets;
fit.Xhat     = fit_param.f_link(get_Xhatinv(decomp,offsets));
if isfield(fit_param,'left_constraint') || isfield(fit_param,'right_constraint')
    fit.left_constraint  = fit_param.left_constraint;
    fit.right_constraint = fit_param.right_constraint;
end
fit.auxiliary=auxiliary;

results = struct();
results.fit = fit;
results.fit_param = fit_param;
results.fit_init  = fit_init;

if opt_fail && (failcnt<failmax)    
    fit_param.failcnt    = failcnt+1;     
    fit_param.rho_offset = 0.8* fit_param.rho_offset;
    fit_param.rho_decomp = 0.8* fit_param.rho_decomp;
    disp(['Optimization Failed. Start With Smaller Step Size: ', num2str(failcnt),'/',num2str(failmax)])
    results = tensor_mrp(Xobs,fit_param,fit_init);
end

end


%% Helpers

function [fit_param, fit_init] = init_all(Xobs, fit_param, fit_init)
% Set/Get default parameters 

% Required Parameters 
if not(isfield(fit_param,'R'));     error('Tensor decomposition order not provided'); end 
if not(isfield(fit_param,'model')); error('Noise Model not provided'); end
if not(isfield(fit_param,'opt'));   error('Optimization Method not provided'); end

% Optional Parameters: iteration, convergence threshold, # fit fail 
if not(isfield(fit_param,'ite_max')); fit_param.ite_max = 2000; end
if not(isfield(fit_param,'disppct')); fit_param.disppct =  5;   end
if not(isfield(fit_param,'convthr')); fit_param.convthr = 1e-12;end
if not(isfield(fit_param,'failcnt')); fit_param.failcnt = 0;    end
if not(isfield(fit_param,'failmax')); fit_param.failmax = 10;   end

% Gradient steps and/or ADAM parameters
if not(isfield(fit_param,'rho_offset')); fit_param.rho_offset = 1e-3*ones(1,fit_param.ite_max); end
if not(isfield(fit_param,'rho_decomp')); fit_param.rho_decomp = 1e-3*ones(1,fit_param.ite_max); end
if strcmp(fit_param.opt,'ADAM') || strcmp(fit_param.opt,'ADAMNC')
   if  not(isfield(fit_param,'beta1')); fit_param.beta1 = 0.9; end
   if  not(isfield(fit_param,'beta2')); fit_param.beta2 = 0.999; end
end

% Multi-linear rank penalty weight (alpha), step (rho_lmultipliers) and aux. weights (rho_mlr)
if not(isfield(fit_param,'alphas'));           fit_param.alphas  = zeros(1,ndims(Xobs)); end
if not(isfield(fit_param,'rho_mlr'));          fit_param.rho_mlr = 1e+1; end
if not(isfield(fit_param,'rho_lmultipliers')); fit_param.rho_lmultipliers = 1e-1*ones(1,fit_param.ite_max); end

% Dimensions to be fitted
if not(isfield(fit_param,'fit_offset_dim')); fit_param.fit_offset_dim = zeros(1,ndims(Xobs)); end
if not(isfield(fit_param,'fit_decomp_dim')); fit_param.fit_decomp_dim =  ones(1,ndims(Xobs)); end
if not(isfield(fit_param,'fit_decomp_lag')); fit_param.fit_decomp_lag =  0;  end
if not(isfield(fit_param,'sprstyc'));        fit_param.sprstyc = zeros(1,ndims(Xobs));end

% Missing data (if all data are observed its simmply one, otherwise, binarry array)
if not(isfield(fit_param,'observed_data')); fit_param.observed_data = 1 ;end

% Get Factor Sizes
[factor_size, fit_param] = get_factor_sizes(Xobs,fit_param);

% Optimizer
opt = fit_param.opt;

% Any Additionnal constraint (left-right constraint or group penalty)
constraint    = zeros(1,ndims(Xobs));
group_penalty = zeros(1,ndims(Xobs));
for dimi = 1:ndims(Xobs)
    if not(all(fit_param.right_constraint{1,dimi}(:)==1)) || not(all(fit_param.left_constraint{1,dimi}(:)==1))
        constraint(1,dimi) = 1;
    end
    
    if isfield(fit_param,'Pg')
        if not(isempty(fit_param.Pg{1,dimi}))
            constraint(1,dimi) = 1;
            group_penalty(1,dimi) = 1;
        end
        
        if not(isfield(fit_param,'lambdag'))
            warning('lambdag not provided. Set to 0.')
        end
    end
end

if (strcmp(opt,'NEWTON') && (sum(constraint)>0))
    warning(['Dimmensions ', num2str(find(constraint)), ' cannot use Newton-step because of group penalty. ADAMNC step instead.' ])
end

fit_param.constraint    = constraint;
fit_param.group_penalty = group_penalty;

% Initialize moments (if ADAM,ADAMNC or Constrained NEWTON ), factors and offsets if not provided
if nargin < 3
    factors   = init_factors(factor_size);
    auxiliary = init_auxiliary(factor_size);
    
    offsets = init_offsets(size(Xobs),fit_param.fit_offset_dim);
    if (contains(opt,'ADAM') || (strcmp(opt,'NEWTON') && (sum(constraint)>0)))
        fit_param.moments = init_moments(factor_size);
    end
    fit_init = struct();  
    fit_init.factors   = factors;
    fit_init.offsets   = offsets;
    fit_init.auxiliary = auxiliary;
else
    
    % Init Factors
    if not(isfield(fit_init,'factors')); factors = init_factors(factor_size); fit_init.factors=factors; end
    
    % Init Auxiliary variables and lagrange multipliers
    if not(isfield(fit_init,'auxiliary')); auxiliary = init_auxiliary(factor_size); fit_init.auxiliary=auxiliary; end
    
    % Init Offset
    if not(isfield(fit_init,'offsets')); offsets = init_offsets(size(Xobs),fit_param.fit_offset_dim); fit_init.offsets=offsets; end
    
    % Init Moments
    if (contains(opt,'ADAM')|| (strcmp(opt,'NEWTON') && (sum(constraint)>0)))
        if isfield(fit_init,'moments'); fit_param.moments = fit_init.moments;
        else; fit_param.moments = init_moments(factor_size); end
    end
end

% Get link functions
[fit_param.F_link, fit_param.f_link, fit_param.df_link ] = get_f_links(fit_param.model);


end

function factors = init_factors(Adims)
% Init the Polyadic Factors
factors = cell(1,length(Adims));

for dimi=1:length(Adims)
    factors{1,dimi} =0.01*randn(Adims(1,dimi),Adims(2,dimi));
end

end

function offsets = init_offsets(Xdims, fit_offset_dim)
% Init Offset
offsets_tmp = 0.01*randn(fit_offset_dim.*Xdims+not(fit_offset_dim));
offsets     = repmat(offsets_tmp, fit_offset_dim + not(fit_offset_dim).*Xdims);
end

function auxiliary = init_auxiliary(Adims)
% Init the augmented Lagragian auxiliary variables (1) and multipliers (2)
auxiliary = cell(2,length(Adims));

for dimi=1:length(Adims)
    auxiliary{1,dimi} =0.01*randn(Adims(1,dimi),Adims(2,dimi)); % Dk
    auxiliary{2,dimi} =0.01*randn(Adims(1,dimi),Adims(2,dimi)); %Zk
end

end

function moments = init_moments(Adims)
% Init ADAM moments to 0
mt  = cell(1,length(Adims));
vt  = cell(1,length(Adims));
gt  = cell(1,length(Adims));

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
% Get Polyadic Factor sizes and normalize the left & right constraints if necessary

if not(isfield(fit_param,'left_constraint')) && not(isfield(fit_param,'right_constraint'))
    % No linear Constraint on Factors
    factor_size1 = size(Xobs);
    factor_size2 = fit_param.R*ones(1,ndims(Xobs));
    
    left_constraint  = num2cell(ones(1,length(factor_size1))); fit_param.left_constraint  = left_constraint;
    right_constraint = num2cell(ones(1,length(factor_size2))); fit_param.right_constraint = right_constraint;
    
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
        left_constraint = num2cell(ones(1,length(factor_size1))); fit_param.left_constraint = left_constraint;
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
        right_constraint = num2cell(ones(1,length(factor_size2))); fit_param.right_constraint = right_constraint;
    end

end

factor_sizes = [factor_size1;factor_size2];

end

function Loss = get_loss(Xobs,Xhatinv,F_link,observed_data)
% For expFam models Loss = - logL
if observed_data==1 
    Loss = (F_link(Xhatinv) - Xobs(:)'*Xhatinv(:));
else
    Loss = (F_link(Xhatinv(find(observed_data))) - Xobs(find(observed_data(:)))'*Xhatinv(find(observed_data(:))));
end
    
end

function mlr_penalty = get_mlr_penalty(auxiliary, factors, fit_param, ZsNNs)
% Get Multilinear rank penalty term
loss_mlr_D  = sum(cell2mat(cellfun(@(x,y,z) x(:)'*(y(:)-z(:)),auxiliary(1,:),factors,auxiliary(2,:) , 'UniformOutput',false)));
loss_mlr_Z  = sum(fit_param.rho_mlr*cell2mat(cellfun(@(x,y) (x(:)-y(:))'*(x(:)-y(:)),factors,auxiliary(2,:) , 'UniformOutput',false)));
loss_mlr_Zn = (fit_param.alphas)*ZsNNs';

mlr_penalty = loss_mlr_D+loss_mlr_Z+loss_mlr_Zn;

end

function Xhatinv = get_Xhatinv(What,Vhat)
    Xhatinv = What + Vhat;
end











