function [GCPnext, What,fit_param] = get_gcp_update(Xobs,What,Vhat,GCP,fit_param,ite)
% Update GCP decomposition of Xobs using: ADAM,ADAMNC or second order method

if nargin < 6;ite = 1; end

% Init Updated GCP Decomposition
GCPnext = GCP;

Xdims   = size(Xobs);
rho_t   = fit_param.rho_decomp(1,ite); % Step
opt     = fit_param.opt;          % Optimizer 'ADAM', 'ADAMNC', 'NEWTON'
f_link  = fit_param.f_link;       % Link function
df_link = fit_param.df_link;      % dLink function
sprstyc = fit_param.sprstyc;      % Sparsity constraint

grpPenalty    = isfield(fit_param,'Pg');
grpPenalty_i  = zeros(1,length(Xdims)); 
if grpPenalty
    lambdag = fit_param.lambdag;
    
    for dimi=1:length(Xdims)
        grpPenalty_i(1,dimi) = not(isempty(fit_param.Pg{1,dimi}));
    end
end


% ADAM Parameters
if not(strcmp(opt,'NEWTON'))
    beta1 = fit_param.beta1;
    beta2 = fit_param.beta2;
    
    mt = fit_param.moments.mt;
    vt = fit_param.moments.vt;
    gt = fit_param.moments.gt;
end

for dimi=1:length(Xdims )
    if fit_param.fit_decomp_dim(1,dimi)
        
        % X ~ [|A1,... An|] : X_(i) = Ai x Zi = Ai x (o Aj)
        
        % i-th GCP component
        Ai = GCPnext{1,dimi};
        
        % Grasp other dimensions
        not_dimi = 1:length(Xdims);
        not_dimi(dimi)=[];
        
        % Khatri-Rao Product
        not_dimi = fliplr(not_dimi);
        Zi = KhatriRaoProd(GCPnext{1,not_dimi});
        
        Xhatiinv = tensor_unfold(What+Vhat,dimi);
        Xhati = f_link(Xhatiinv);
        
 
        Xobsi = tensor_unfold(Xobs,dimi);
        dFi   = Xhati - Xobsi;
        
        

         % Gradient with respect to Ai
        if grpPenalty_i(1,dimi)
            indicator = fit_param.Pg{1,dimi};
            [grad_group_penalty,hess_penalty] = get_ggrad_penalty(Ai,lambdag,indicator);  
        else
            grad_group_penalty = 0;
            hess_penalty = zeros(size(Ai));
        end
        
        

        % Gradient with respect to Ai
        dFdAi = dFi*Zi + sprstyc(1,dimi)*Ai + grad_group_penalty ;
        
     
         
        
        gt{1,dimi} = dFdAi;
        
        if strcmp(opt,'NEWTON')
           
            dfi  = df_link(Xhatiinv);           
            dAi  = zeros(size(dFdAi));
            
   
            
            % Row-wize Update of dAi
            for di=1:size(dFdAi,1)
                % Ji = Zi' x Diag(dfi(di,:)) x Zi + etai x I
                Ji = bsxfun(@times,Zi',dfi(di,:)) * Zi + (sprstyc(1,dimi)+1e-12) * eye(size(Ai,2));
                dAitmp = dFdAi(di,:) /(Ji+diag(hess_penalty(di,:)));
                dAi(di,:) = dAitmp;

            end
            
            
            
            
            
        else
            if strcmp(opt,'ADAM')
                
                % Fixed betas
                beta1c=beta1;
                beta2c=beta2;
                
                % Moment Update
                mt{1,dimi} = beta1c*mt{1,dimi} + (1-beta1c)*gt{1,dimi};
                vt{1,dimi} = beta2c*vt{1,dimi} + (1-beta2c)*gt{1,dimi}.^2;
                
                % Bias correction
                mtt = mt{1,dimi}./(1-beta1.^ite);
                vtt = vt{1,dimi}./(1-beta2.^ite);
                
            elseif strcmp(opt,'ADAMNC')
                
                % Varying betas
                beta1c = beta1*0.9^(ite-1);
                beta2c = 1-1/ite;
                
                % Moment Update
                mt{1,dimi} = beta1c*mt{1,dimi} + (1-beta1c)*gt{1,dimi};
                vt{1,dimi} = beta2c*vt{1,dimi} + (1-beta2c)*gt{1,dimi}.^2;
                
                % No Bias correction
                mtt = mt{1,dimi};
                vtt = vt{1,dimi};
            end
            
            % ADAM(NC) update
            dAi = mtt./(sqrt(vtt)+1e-10);
            
        end
        
        Ai = Ai - rho_t*dAi;
        GCPnext{1,dimi}  = Ai;
        What = tensor_fold(Ai*Zi',Xdims,dimi);
        
        
    end
    
    if not(strcmp(opt,'NEWTON'))
        fit_param.moments.mt=mt;
        fit_param.moments.vt=vt;
        fit_param.moments.gt=gt;
    end
    
    
end
end

