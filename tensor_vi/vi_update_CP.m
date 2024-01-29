function vi_var = vi_update_CP(Xobs, vi_var,vi_param)
% Variational update of the CP factors

% Grasp Current Variational Parameters
CP_mean     = vi_var.CP_mean;
CP_variance = vi_var.CP_variance;
Ulatent = vi_var.latent_mean;
Voffset = vi_var.offset_mean;
Eshape  = vi_var.shape;

% Dimension to be updated
update_CP_dim = vi_param.update_CP_dim;

% Use gpuArrays
use_gpu  = vi_param.use_gpu;

% Sparsity / Block Sparsity
use_sparse = vi_param.sparse;
if strcmp(use_sparse, 'block-sparse')
    indices_block = vi_param.indices_block;
end

% Priors
CP_prior_mean      = vi_var.CP_prior_mean;
CP_prior_precision = vi_var.CP_prior_precision;

% Problem size
Xdims = size(Xobs);

% Tensor Rank
R = size(CP_mean{1},2);

% Diagonal Elements for Precision
RR = (0:(R-1))*(R+1)+1;

% Deal with missing data
observed_data = vi_param.observed_data;
 
% Full observed_data or few missing
for dimn = 1:length(CP_prior_mean)

    % Loop on dimensions: n-th Unfoldings
    if update_CP_dim(dimn)
        if strcmp(use_sparse, 'false')
            
            % Pseudo Variable
            Z  =  ((Xobs-Eshape)/2 - Voffset.*Ulatent).*observed_data;
            
            % <B'UB>
            BUZ = mttkrp_custom(Z, CP_mean, dimn);
            
            % <B'><U><Z>
            BUB = mttkrp_custom(Ulatent, get_AAt(CP_mean,CP_variance), dimn);

        elseif strcmp(use_sparse, 'block-sparse')
            
            % Pseudo Variable
            Z = cellfun(@(X,V,U,O) ((X-Eshape)/2 - V.*U).*O, ...
                Xobs, Voffset, Ulatent, observed_data, 'UniformOutput', false);
            
            % <B'><U><Z>
            BUZ = mttkrp_block(Z, indices_block , CP_mean, dimn);
            
            % <B'UB>
            BUB = mttkrp_block(Ulatent, indices_block, get_AAt(CP_mean,CP_variance), dimn);
         
        elseif strcmp(use_sparse, 'sparse')
            
            % Pseudo Variable
            Z  = ((Xobs-Eshape)/2 - Voffset.*Ulatent).*observed_data;
            
            % <B'><U><Z>
            BUZ = mttkrp_sparse(Z, indices_sparse , CP_mean, dimn);
            
            % <B'UB>
            BUB = mttkrp_sparse(Ulatent,indices_block,  get_AAt(CP_mean,CP_variance), dimn);
            
        else
            error('Sparsity mode not implemented')
        end
        
        % Priors
        prec_prior = CP_prior_precision{1,dimn};
        mean_prior = CP_prior_mean{1,dimn};
        
        % Temporary Update (precisions are diag)
        prec_post = prec_prior+BUB;
        mean_post_tmp = BUZ + prec_prior(:,RR).*mean_prior;
        
        % Invert Precision and Update mean
        if not(use_gpu)
            for dimi = 1:size(BUZ,1)
                
                vari_post_i = inv(reshape(prec_post(dimi,:),[R,R]));
                mean_post_i = vari_post_i*mean_post_tmp(dimi,:)';
                vari_post_i = vari_post_i(:)';
                
                CP_mean{1,dimn}(dimi,:) = mean_post_i;
                CP_variance{1,dimn}(dimi,:) = vari_post_i;
                
            end
        else
            
            prec_post = reshape(prec_post',R,R,[] );
            mean_post_tmp = reshape(mean_post_tmp',R,1,[] );
            
            vari_post = pagefun(@inv,prec_post);
            mean_post = pagefun(@mtimes,vari_post, mean_post_tmp);
            
            CP_mean{1,dimn} = reshape(mean_post, R, [])';
            CP_variance{1,dimn} = reshape(vari_post, R*R, [])';
            
        end     
    end
end

% Reconstruct the First and Second moment of the low rank tensor
if strcmp(use_sparse, 'false')
    tensor_mean = tensor_reconstruct(CP_mean);
    AAt = get_AAt(CP_mean,CP_variance);
    tensor2_mean   = tensor_reconstruct(AAt);

elseif strcmp(use_sparse, 'block-sparse')
    tensor_mean = tensor_reconstruct_block(CP_mean, indices_block);
    AAt = get_AAt(CP_mean,CP_variance);
    tensor2_mean   = tensor_reconstruct_block(AAt, indices_block);
    
end

% Save Posterior
vi_var.CP_mean = CP_mean;
vi_var.CP_variance = CP_variance;

% Save tensor moments
vi_var.tensor_mean  = tensor_mean;
vi_var.tensor2_mean = tensor2_mean;


end




