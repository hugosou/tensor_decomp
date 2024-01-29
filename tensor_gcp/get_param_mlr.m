function [init_param,fit_param] = get_param_mlr(Xobscur,fit_param_tmp)

fit_param = fit_param_tmp;
fit_param.fit_offdim = fit_param_tmp.fit_offdim(1,1:length(size(Xobscur)));

Winit  = 0.001*randn(size(Xobscur));
vinit  = 0*rand((fit_param.fit_offdim).*size(Winit)+not(fit_param.fit_offdim));
Vinit  = repmat(vinit, fit_param.fit_offdim + not(fit_param.fit_offdim).*size(Winit));
Zsinit = 1*rand([size(Winit),length(size(Winit))]);
Dsinit = rand([size(Winit),length(size(Winit))]);

% Build Init Params
init_param = struct();
init_param.W  = Winit;
init_param.V  = Vinit;
init_param.Zs = Zsinit;
init_param.Ds = Dsinit;


fit_param.lambda     = fit_param.lambda(1,1:length(size(Xobscur)));
fit_param.fit_offdim = fit_param.fit_offdim(1,1:length(size(Xobscur)));

end