function vi_var = vi_update_offset(Xobs, vi_var,vi_param)
% Variational update of the offset tensor constrained along fit_offset_dim

% Grasp Current Posteriors
Wtensor = vi_var.tensor_mean;
Ulatent = vi_var.latent_mean;
Eshape  = vi_var.shape;

% Prior Parameters
offset_prior_mean = vi_var.offset_prior_mean;
offset_prior_prec = vi_var.offset_prior_precision;

fit_offset_dim = vi_param.fit_offset_dim;
observed_data  = vi_param.observed_data;

if strcmp(vi_param.sparse, 'false') || strcmp(vi_param.sparse, 'sparse')
    
   [offset_posterior_mean, offset_posterior_vari] = ...
    update_offset(offset_prior_mean, offset_prior_prec, ...
    observed_data, fit_offset_dim, Xobs, Wtensor, Ulatent, Eshape);    
    
elseif strcmp(vi_param.sparse, 'block-sparse')
     
    num_blocks = size(Wtensor,1);
    offset_posterior_mean = cell(num_blocks,1);
    offset_posterior_vari = cell(num_blocks,1);
     
    for block =1:num_blocks
        [offset_posterior_mean{block}, offset_posterior_vari{block}] = ...
            update_offset(...
            offset_prior_mean{block}, offset_prior_prec{block}, ...
            observed_data{block}, fit_offset_dim, Xobs{block}, ...
            Wtensor{block}, Ulatent{block}, Eshape);
    end
end
  
% Update current sample
vi_var.offset_mean     = offset_posterior_mean;
vi_var.offset_variance = offset_posterior_vari;

end

function [offset_posterior_mean, offset_posterior_vari] = ...
    update_offset(offset_prior_mean, offset_prior_prec, ...
    observed_data, fit_offset_dim, Xobs, Wtensor, Ulatent, Eshape)

% Using only observed data to update offset
Ulatent = Ulatent.*observed_data;

% Observation Tensor Size
Xdims = size(Xobs);

% Constrained Dimensions
non_fit_offset_dim = find(not(fit_offset_dim));

% Auxiliazry Variable Used in the augmented Gaussian Likelihood
Ztmp = (Xobs-Eshape)/2 - Wtensor.*Ulatent;
Ztmp = Ztmp.*observed_data;

% Compact dimensions
Xdims_red = [Xdims(find(fit_offset_dim)), 1];

% Matlab annoyingly reshape tensor of last dim 1
if length(Xdims) < length(fit_offset_dim)
    Xdims = [Xdims, ones(1,length(fit_offset_dim)-length(Xdims))];
end
 
% Compact auxiliary
Utilde = reshape(sum(Ulatent,non_fit_offset_dim), Xdims_red);
Ztilde = reshape(sum(Ztmp,non_fit_offset_dim), Xdims_red)./(Utilde+eps);

% Update offset mean and variance using prior mean and precision
offset_posterior_vari = 1./(Utilde+offset_prior_prec);
offset_posterior_mean = offset_posterior_vari.*(Utilde.*Ztilde + offset_prior_prec.*offset_prior_mean);

offset_posterior_vari= reshape(offset_posterior_vari , fit_offset_dim.*Xdims+not(fit_offset_dim));
offset_posterior_mean= reshape(offset_posterior_mean , fit_offset_dim.*Xdims+not(fit_offset_dim));

% Repeat Offset along constrained dimensions
repeat_value = not(fit_offset_dim).*Xdims + fit_offset_dim;
offset_posterior_vari = repmat(offset_posterior_vari,repeat_value);
offset_posterior_mean = repmat(offset_posterior_mean,repeat_value);

end
 





