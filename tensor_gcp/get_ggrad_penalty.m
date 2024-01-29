function [grad_penalty,hess_penalty] = get_ggrad_penalty(factor,lambda,indicator)
% The size of lambda is  G x R (or 1 x G)

G = size(lambda,1);
R = size(lambda,2);

if R>1
    indicator = indicator';
     
    hess_penalty = indicator'*(lambda.*(indicator*(factor.*factor+1e-10)).^(-1/2)); % Not exactly the Hessian
    grad_penalty = hess_penalty.*factor;
else
    lambda = lambda';
    trAPPA = (diag(factor*factor')'*indicator);
    trAPPA = lambda./(sqrt(trAPPA)+eps);
    hess_penalty = diag(sum(trAPPA.*indicator,2));
    grad_penalty = hess_penalty*factor;
   
end
    
end



