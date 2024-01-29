function [Aknext, What, DZknext, fit_param,Nhats,ZsNNs] = get_mrp_update(Xobs,What,Vhat,Ak,DZk,fit_param,ite)
   

  
if nargin < 6; ite = 1;end

% Linear constraints on factors
left_constraint  = fit_param.left_constraint;
right_constraint = fit_param.right_constraint;

% Missing data (=1 if none)
observed_data = fit_param.observed_data;

% Current decomposition
Aknext  = Ak;
DZknext = DZk;


Xdims   = size(Xobs);
rho_t   = fit_param.rho_decomp(1,ite); % Step
opt     = fit_param.opt;               % Optimizer 'ADAM', 'ADAMNC', 'NEWTON'
f_link  = fit_param.f_link;            % Link function
sprstyc  = fit_param.sprstyc;          % Sparsity constraint

% MLR Params
rho_mlr   = fit_param.rho_mlr;         % Rho
rho_lmultipliers = fit_param.rho_lmultipliers(1,ite); % Rho
alphas    = fit_param.alphas;   % Rank penalty

% Factors with a left-right constraint or group penalty
constraint     = fit_param.constraint;
group_penalty  = fit_param.constraint;

% ADAM/ADAMNC Parameters
if  contains(opt,'ADAM') || (strcmp(opt,'NEWTON') && (sum(constraint)>0))
    beta1 = fit_param.beta1;
    beta2 = fit_param.beta2;
    
    mt = fit_param.moments.mt;
    vt = fit_param.moments.vt;
    gt = fit_param.moments.gt;
end


for dimi=1:length(Xdims)
    if fit_param.fit_decomp_dim(1,dimi)
        % For W ~ [|A1,... An|] : W_(i) = Ai x Zi = Ai x (o Aj)
        
        % i-th GCP component
        Ai = Aknext{1,dimi};
        Di = DZknext{1,dimi};
        Zi = DZknext{2,dimi};
        
        % Full factors
        Ck = get_constrained_factor(Aknext, fit_param);
        
        % Grasp other dimensions
        not_dimi = 1:length(Xdims);
        not_dimi(dimi)=[];
        
        % Khatri-Rao Product
        not_dimi = fliplr(not_dimi);
        Ani = KhatriRaoProd(Ck{1,not_dimi});

        % Gradient of the log likelihood
        Xhati = tensor_unfold(What+Vhat,dimi);
        Xobsi = tensor_unfold(Xobs,dimi);        
        dFdA0 = (tensor_unfold(observed_data,dimi).*(f_link(Xhati) - Xobsi)) * Ani; % TO  BE TESTED
        
        % Group Penalty on the Factor
        if group_penalty(1,dimi)
            Ai_augmented = left_constraint{1,dimi}*Ai*right_constraint{1,dimi};
            grad_group_penalty = get_ggrad_penalty(Ai_augmented,fit_param.lambdag,fit_param.Pg{1,dimi});  
            dFdA0 = dFdA0 + grad_group_penalty;
        end
        
        % Gradient with : linear constraint, group and sparsity penalty
        dFdAi = left_constraint{1,dimi}' * dFdA0 * right_constraint{1,dimi}' + sprstyc(1,dimi)*Ai;
        
        % Add MLR constraint
        if alphas(dimi)>eps
            dFdAi =  dFdAi + rho_mlr * (Ai-Zi) + Di;
        end
        
        % Save gradient
        
        
        
        if strcmp(opt,'GRAD')
            % Gradient descent
            Ai = Ai - rho_t*dFdAi;
            
        elseif  contains(opt,'ADAM') || (constraint(1,dimi))
            % ADAM/ADAMNC update
            gt{1,dimi}   = dFdAi;
            [dAi,mt{1,dimi},vt{1,dimi},gt{1,dimi}] = adam_step(opt,beta1,beta2,mt{1,dimi},vt{1,dimi},gt{1,dimi},ite);
            Ai = Ai - rho_t*dAi;
            
        elseif strcmp(opt,'NEWTON') && not(constraint(1,dimi))
            % Second order method update
            dAi = newton_step(Xhati,dFdAi,Ani,sprstyc(1,dimi),rho_mlr,fit_param.df_link);
            Ai = Ai - rho_t*dAi;
        else
            error('Optimizer not supported')
            
        end
        
        % Update factor
        Aknext{1,dimi}  = Ai;
        
        % Sanity check
        Ai(isnan(Ai)) = 0;
        Ai(isinf(Ai)) = 0;
        
        % Update decomposition
        What = tensor_fold(left_constraint{1,dimi}*Ai*right_constraint{1,dimi}*Ani',Xdims,dimi);
        
        % Update Zi and Di if MLR constraint
        if alphas(dimi)>0
           Zi_target = Ai + 1/rho_mlr*Di;
           [P1,Sig,P2] = svd(Zi_target,'econ');
           
           Sig_tmp = Sig-(alphas(dimi)/rho_mlr);
           Sig_tmp(1,1) = max(eps, Sig_tmp(1,1)); % Make sure at least one component.
           
           Sig_thr = Sig_tmp.*(Sig_tmp>0);
           
           Zi = P1*Sig_thr*P2';
           
           DZknext{2,dimi} = Zi;
           Nhats(1,dimi) = sum(diag(Sig_tmp)>0);
           ZsNNs(1,dimi) = sum(diag(Sig_thr));
           
           % Update Di
           DZknext{1,dimi} = Di + rho_lmultipliers*(Ai - Zi); % Add ADAM STEP ?
           
        else
           Nhats(1,dimi) = min(size(Ai));
           ZsNNs(1,dimi) = 0;
        end
      
        

        
    end
    
    
    % ADAM/ADAMNC Parameters
    if  contains(opt,'ADAM')
        fit_param.moments.mt=mt;
        fit_param.moments.vt=vt;
        fit_param.moments.gt=gt;
    end
    
    
end

end


%% Helpers
function dAi = newton_step(Xhati,dFdAi,Ani,sprstyci,rho_mlr,df_link)

if nargin < 3
    sprstyci = 0;
end

if nargin < 4
    rho_mlr = 0;
end

dfi  = df_link(Xhati);
dAi  = zeros(size(dFdAi));

% Row-wise update (faster empirically)
for di=1:size(dFdAi,1)
    % Ji = Zi' x Diag(dfi(di,:)) x Zi + etai x I
    Ji = bsxfun(@times,Ani',dfi(di,:)) * Ani + (sprstyci+rho_mlr+1e-12) * eye(size(dFdAi,2));
    dAitmp = dFdAi(di,:) /Ji;
    dAi(di,:) = dAitmp;
    
end

end



function [dA,mt,vt,gt] = adam_step(opt,beta1,beta2,mt,vt,gt,ite)

if strcmp(opt,'ADAM')
    
    % Fixed betas
    beta1c=beta1;
    beta2c=beta2;
    
    % Moment Update
    mt = beta1c*mt + (1-beta1c)*gt;
    vt = beta2c*vt + (1-beta2c)*gt.^2;
    
    % Bias correction
    mtt = mt./(1-beta1.^ite);
    vtt = vt./(1-beta2.^ite);
    
else 
    % Default is ADAMNS
    % Varying betas
    beta1c = beta1*0.9^(ite-1);
    beta2c = 1-1/ite;
    
    % Moment Update
    mt = beta1c*mt + (1-beta1c)*gt;
    vt = beta2c*vt + (1-beta2c)*gt.^2;
    
    % No Bias correction
    mtt = mt;
    vtt = vt;
    
end

vt(isnan(vt)) = 1;
vt(isinf(vt)) = 1;
mt(isnan(mt)) = 0;
mt(isinf(mt)) = 0;

dA = mtt./(sqrt(vtt)+1e-10);

end







