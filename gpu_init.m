function [Xobs,vi_param,vi_var] = gpu_init(Xobs,vi_param,vi_var)

% Cast Variable of interest as gpuArray
if isfield(vi_param, 'use_gpu')
    if vi_param.use_gpu
        % Setup precisions
        %ClassVar = 'double'; 
        ClassVar = 'single';
        
        
        % Observed tensor
        Xobs = gpuArray(cast(Xobs,ClassVar));
        
        % Observed / Missing data
        vi_param.observed_data = ...
            gpuArray(cast(vi_param.observed_data,ClassVar));
        
        % Fit params
        vi_var_names = fieldnames(vi_var);
        for parami = 1:length(vi_var_names)            

            vi_var_cur = eval(['vi_var.' ,vi_var_names{parami},';']);
            
            % Factors
            if isa(vi_var_cur,'cell')
                vi_var_cur = cellfun(@(Z) gpuArray(cast(Z,ClassVar)), ...
                    vi_var_cur, 'UniformOutput', false);
                
                % Others
            elseif isa(vi_var_cur,'double') || isa(vi_var_cur,'single')
                vi_var_cur = gpuArray(cast(vi_var_cur,ClassVar));
            end
            
            eval(['vi_var.' ,vi_var_names{parami},'= vi_var_cur;']);
            
        end
    end
end

end