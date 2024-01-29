function offset_next = get_offset_update(X,W,offset,fit_param,ite)
% GET_OFFSET_UPDATE Updates the offset such that E(X) = f_link(W+offset)
% X, W and offset are d1xd2x...xdN tensors
% Inputs: X observation tensor
%         W low rank dynamics tensor
%         offset tensor
%         fit_param is a structure containing:
%               f_link ExpFam link function
%               df_link its piecewize derivative
%               rho_o gradient step
%               fit_dims 1xN boolean giving the true dimension of V
%               (note: the bigger fit_dims the smaller should be rho_V)
%               missing data: 1 if note, binary array otherwise


fit_offset_dim = fit_param.fit_offset_dim;
observed_data   = fit_param.observed_data;

f_link  = fit_param.f_link;
df_link = fit_param.df_link;


if nargin<5; ite = 1; end
rho_o = fit_param.rho_offset(1,ite);

Wdims = size(W);
if not(length(fit_offset_dim)==length(Wdims))
    error('Incorect offset fitted dimensions')
end

not_fit_dims = find(not(fit_offset_dim));
repeat_value = not(fit_offset_dim).*Wdims + fit_offset_dim;



% Gradient and Hessian
dV   = observed_data.*(f_link(W + offset)-X); 
ddV  = df_link(W + offset);


% [N,T,L,K] = size(X);
% tmp = repmat(eye(L),[1,T*K]);
% dV1 = tensor_unfold(dV,1);
% vvc = squeeze(mean(mean(offset,2),4));
% dvv = dV1*tmp';
% vvc = vvc-0.0001*rho_o*dvv;
% offset_next = tensor_fold(vvc*tmp,[N,T,L,K],1);


if not(isempty(not_fit_dims))
    dV = sum(dV,not_fit_dims);
    ddV = (1./(1e-10+sum(ddV,not_fit_dims))).*dV;
    ddV(isinf(ddV)|isnan(ddV)) = eps;
end

% Second Order
ddV = repmat(ddV,repeat_value);
offset_next = offset - rho_o*ddV;


end




