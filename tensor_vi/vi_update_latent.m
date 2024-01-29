function vi_var = vi_update_latent(Xobs, vi_var,vi_param)
% Variational update of the PG distributed latent 

% Current Shape
Eshape = vi_var.shape;

if strcmp(vi_param.sparse, 'false') || strcmp(vi_param.sparse, 'sparse')
    
    % First moments
    tensor =  vi_var.tensor_mean;
    offset =  vi_var.offset_mean;
    
    % Second moments
    tensor2 = vi_var.tensor2_mean;
    offset2 = vi_var.offset_variance + vi_var.offset_mean.^2;
    
    observed_data = vi_param.observed_data;
    if all(observed_data(:)==1)
        % All data are observed
        observed_id = (1:numel(Xobs))';
    else
        % There is missing data.
        observed_id = find(observed_data);
    end
    
    latent = update_latent(tensor, tensor2, offset, offset2, ...
        observed_id, Xobs, Eshape);
    
elseif strcmp(vi_param.sparse, 'block-sparse')
    
    num_blocks = size(vi_var.tensor_mean,1);
    latent = cell(num_blocks,1);
    
    for block =1:num_blocks
        
        % First moments
        tensor =  vi_var.tensor_mean{block};
        offset =  vi_var.offset_mean{block};
         
        % Second moments
        tensor2 = vi_var.tensor2_mean{block};
        offset2 = vi_var.offset_variance{block} ...
            + vi_var.offset_mean{block}.^2;
        
        observed_data = vi_param.observed_data{block};
        if all(observed_data(:)==1)
            % All data are observed
            observed_id = (1:numel(Xobs{block}))';
        else
            % There is missing data.
            observed_id = find(observed_data);
        end
        
        latent{block} = update_latent(tensor, tensor2, offset, offset2, ...
            observed_id, Xobs{block}, Eshape);
        
    end
end

% Update variables
vi_var.latent_mean = latent;

end


function latent = update_latent(tensor, tensor2, offset, offset2, ...
    observed_id, Xobs, Eshape)

% sqrt(<(W +V)^2>)
omega = sqrt(offset2(observed_id) + tensor2(observed_id)...
    + 2.*tensor(observed_id).*offset(observed_id));
Xtmp  = Xobs(observed_id);

% Update only relevant means
latent = zeros(numel(Xobs),1);
latent_mean_tmp = ((Eshape + Xtmp)./(2*omega)) .*tanh(omega/2);
latent(observed_id) = latent_mean_tmp;

% Reshape Latents
latent = reshape(latent, size(Xobs));

end



