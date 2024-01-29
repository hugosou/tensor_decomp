function vi_var = tensor_variational_inference(Xobs,vi_param,vi_var)
% Bayesian tensor CP decomposition of count data 
% Inference via Variational Inference and Polya-Gamma  augmentation 
% @Hugo.S

if not(isfield(vi_param,'ite_max')); error('ite_max required field'); else; ite_max = vi_param.ite_max; end
if not(isfield(vi_param,'shape_update')); vi_param.shape_update = 'MM-G'; end
if not(isfield(vi_param,'disppct')); vi_param.disppct = 1; end

% Optional : Automatic Relevance Determination Parameter
if not(isfield(vi_param,'dim_neuron')); vi_param.dim_neuron = 0; end
if not(isfield(vi_param,'shared_precision_dim')); vi_param.shared_precision_dim = 0; end

% Optional : Missing Data
if not(isfield(vi_param,'observed_data'));   vi_param.observed_data = 1; end

% Optional : Dimensions to fit
if not(isfield(vi_param,'update_CP_dim')); vi_param.update_CP_dim  = ones(1,ndims(Xobs)); end
if not(isfield(vi_param,'fit_offset_dim'));vi_param.fit_offset_dim = ones(1, ndims(Xobs)); end
if not(isfield(vi_param,'fit_shape'));     vi_param.fit_shape = 1; end

% Init variational variables
if nargin <3; vi_var = struct(); end
[Xobs, vi_var,vi_param] = vi_init(Xobs,vi_var, vi_param);

loss_tot  = zeros(ite_max,1);
shape_tot = zeros(ite_max,1);

ref = vi_var.CP_mean;
precn = max(1+floor(log10(ite_max)),1);
logger = ['Iterations: %.', num2str(precn),'d/%d %s%.10g %s%.5g \n'];


for ite=1:ite_max
    % Variational Update: Latent U
    vi_var = vi_update_latent(Xobs, vi_var,vi_param);
    
    % Variational Update: CP factors
    vi_var = vi_update_CP(Xobs, vi_var,vi_param);
    
    % Variational Update: Offset V
    if sum(vi_param.fit_offset_dim)>0
        vi_var = vi_update_offset(Xobs, vi_var,vi_param);
    end
      
    % Variational Update: Precision for ARD
    if sum(vi_param.shared_precision_dim)>0
        vi_var = vi_update_precision_shared(vi_var,vi_param);
    end
     
    % Variational Update: Precision with Neuron Groups
    if vi_param.dim_neuron>0
        vi_var = vi_update_precision_mode(vi_var,vi_param);
    end

    % Variational Update: Shape
    if vi_param.fit_shape>0
        vi_var = vi_update_shape(Xobs, vi_var,vi_param);
    end
    
    % Display Loss 
    [loss,loss_str,ref] = get_loss(vi_var,vi_param,ref);
    display_loss = mod(100 * ite/ite_max,vi_param.disppct)==0 && vi_param.disppct>0;
    if display_loss
        fprintf(logger,ite,ite_max,loss_str,loss,'| Shape = ',vi_var.shape); 
    end
    
    loss_tot(ite)  = loss;
    shape_tot(ite) = vi_var.shape;
    
end

vi_var.loss_tot  = loss_tot; 
vi_var.shape_tot = shape_tot;
    

end


function [loss,loss_str,ref] = get_loss(vi_var,vi_param,ref)
if contains(vi_param.shape_update,'MM') || strcmp(vi_param.shape_update,'numerical')
    loss_str = '| FE = ';
    loss = vi_var.FE;
else
    loss_str = '| dSim = ';
    loss = get_similarity({ref, vi_var.CP_mean}, 1); ref = vi_var.CP_mean;
    loss = loss(2);
end
end



