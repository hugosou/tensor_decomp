function xopt = fmin_newton(df,ddf,x0,ite_max,tol)
% 2nd order minimization method well suited forfast GPU computing
% Constrain x to be positive and exploit f's convexity

if nargin < 5
   tol = 1e-10; 
end

if nargin <4
   ite_max = 1000; 
end

% Init
xcur= x0;

for ite=1:ite_max
    xnew = max(xcur - df(xcur) / (ddf(xcur)+eps), 0);

    if abs(xnew-xcur)/abs(0.5*xnew+0.5*xcur)< tol
        break;     
    else
        xcur = xnew;
    end
end

xopt = xnew;

end