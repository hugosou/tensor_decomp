%% Load and organize data

%addpath('./Violinplot')
addpath(genpath('./'))
%load('/home/sou/Documents/Data/dresults/xval_mismatch_RSPdRSPgSC_collapsed1_2021_02_07_22_33.mat')
%load('/home/sou/Documents/Data/dresults/xval_standard_RSPdRSPgSC_collapsed1_2021_02_05_14_09.mat')
%load('/home/sou/Documents/Data/dresults/xval_mismatch_RSPdRSPgSC_collapsed1_2021_02_10_21_46.mat')
load('/home/sou/Documents/Data/dresults/xval_standard_RSPdRSPgSCSUBV1_collapsed1_2021_02_10_00_54.mat')

% Add master folders
%addpath(genpath('./tensorfact_master'))

% Machine Path
folder = '/home/sou/Documents/Data/';
data_folder = [folder, 'data_sepi/'];
resu_folder = [folder, 'dresults/'];


% Choose dataset: 'standard', 'mismatch'
dataset_str = 'standard';

% Load PRE-PROCESSED data
if strcmp(dataset_str, 'standard');load([data_folder,'data_sepim']);
elseif strcmp(dataset_str, 'mismatch') ;load([data_folder,'data_sepim_mismatch']);
end

% Keep regions from : 'N/A','RSPd','RSPg','SC','SUB','V1', 'Hip'
kept_region = cellstr(experimental_parameters.dict_region)';

% Use a 3D or 4D tensor
collapse_conditions = experimental_parameters.collapse_conditions;

% Attribute to group neurons: 'ctype','layer','region'
neuron_group_discriminant = 'layer';

% Build Tensor
%data_sepim.record_layer = data_sepim.record_layr;
%data_sepim.record_celltype = data_sepim.record_ctype;
%data_sepim.observed_tensor = ones(size()); 

[Xobs, ~] = data_sepi_load_V1(data_sepim,kept_region,neuron_group_discriminant);
%datam = data_sepi_load(data_sepim,kept_region,neuron_group_discriminant);

%%

% Extract experimental parameters
param_names = fieldnames(experimental_parameters);
for parami = 1:length(param_names)
    eval([param_names{parami} '=experimental_parameters.' param_names{parami},';']);
end





%% Train and Test Deviance of the null Distribution

n_test = floor(size(Xobs,1)/2);   % Number of Leave Neuron Out 
N_test = 4;                       % Number of LNO tests;



D_0_test  = zeros(size(test_folder_outer,1), N_test, n_test);
D_0_train = zeros(size(test_folder_outer,1),1);

D_0_train2 = zeros(size(test_folder_outer,1),1);


for folder_id= 1:size(test_folder_outer,1)
    cur_test_folders  = test_folder_outer(folder_id,:);
    cur_train_folders = train_folder_outer(folder_id,:);
    
    Xobs_train = Xobs(:,:,:,cur_train_folders);
    Xobs_test  = Xobs(:,:,:,cur_test_folders);
    
    X01 = mean(Xobs_train(:));
    X02 = poissrnd(mean(Xobs_train(:)),size(Xobs_train) );
    
    X0train = deviance_poisson(Xobs_train,repmat(X01,size(Xobs_train)));
    X0train2 = deviance_poisson(Xobs_train,X02);
    
    D_0_train(folder_id,1) = X0train;
    D_0_train2(folder_id,1) = X0train2;
    
    
    for nn = 1:N_test
        idtest = perm_tot(nn,:);
        idtrain = 1:size(Xobs_test,1); idtrain(idtest) =[];
        Xtesti = tensor_remove(Xobs_test,1,idtrain);
        Xpredi = repmat(X01,size(Xtesti));
        D = deviance_poisson_nwise(Xtesti, Xpredi);
        D_0_test(folder_id, nn,:) = D;
    end
end

D0tr = median(D_0_train);
D0te = median(D_0_test(:));


%% Gather Deviance and Estimate Similarities

[Nmodels, Nhyper1,Nhyper2,Nhyper3] = size(Dtot{1,1}.Dtrain);
Nfolder = size(Dtot,2);

% Deviances
Dtot_train = zeros([size(Dtot{1,1}.Dtrain), Nfolder]);
Dtot_tests = zeros([size(Dtot{1,1}.Dtest),  Nfolder]);
for nfolder = 1:Nfolder
    Dtot_train(:,:,:,:,nfolder) = Dtot{1,nfolder}.Dtrain;
    Dtot_tests(:,:,:,:,:,:,nfolder) = Dtot{1,nfolder}.Dtest;
end


% Estimate Similarities and reorder models first pass
used_dim = [1,1,1,0]; used_dim = used_dim(1:ndims(Xobs));
[smlty_tot,ref_tot,permt_tot,~,sig_tot] = get_similarity_tot(models_tot,used_dim);

% Reorient/Reorder CPs
models_reordered = reorder_cps(models_tot,permt_tot,sig_tot);

%%
% Trials and conditions miht be collapsed
modelsf = models_reordered;
UU = experimental_parameters.condition_design;
if experimental_parameters.collapse_conditions
    
    for ii = 1:numel(models_tot)
        modelsf{ii}{1,3} = UU * models_reordered{ii}{1,3}./sum(UU,2);
    end
    
    % Update Similarity Metrix
    [smlty_tot,ref_tot,permt_tot,~,sig_tot] =  get_similarity_tot(modelsf,used_dim);
    
    % Reorder/Re-orient
    models_reordered2= reorder_cps(modelsf,permt_tot,sig_tot);
    gathered_cp      = gather_cps(models_reordered2);
else 
    models_reordered2 = models_reordered;
    gathered_cp      = gather_cps(models_reordered); 
end




%% Deviance - Similarity Plots

VOIstr = {'Dtrain','Dtest','Sim-Within'};
hyp2_style = [linspace(0,1,Nhyper2).^2',0.5*linspace(1,0,Nhyper2).^2',linspace(1,0,Nhyper2)'];
hyp3_style = {'-','none','-.',':'};

Dtot_tests2 = reshape(Dtot_tests, [size(Dtot_tests,1:4),prod(size(Dtot_tests,5:7))]);
VOI = {Dtot_train,Dtot_tests2,smlty_tot};
D0I = {D0tr,D0te,1};

for VOIid = 1:size(VOIstr,2)
    
    figure
    pp = [];
    
    VOIcur = VOI{1,VOIid};
    
    for modeli = 1:Nmodels
        p = subplot(1,Nmodels, modeli); hold on
        
        for jj = 1:Nhyper2
            for kk = 1:Nhyper3
                
                Dp = median(squeeze(VOIcur(modeli,:,jj,kk,:)),2);
                
                if VOIid<3
                    Dp = 1 - Dp/D0I{1,VOIid};
                end
                
                Dtp = squeeze(VOIcur(modeli,:,jj,kk,:));
                if VOIid<3
                    Dtp = 1 - Dtp/D0I{1,VOIid};
                end
                
                Dp2  = median(Dtp,2);
                Dtp2 = std(Dtp,0,2);
                
                hyp1 = hyperparams{1,1};
                
                if VOIid==3
                    errorbar(hyp1 , Dp2,Dtp2, 'color', hyp2_style(jj,:),'linestyle',hyp3_style{kk}  , 'linewidth',1.6);
                else
                    plot(hyp1 , Dp, 'color', hyp2_style(jj,:),'linestyle',hyp3_style{kk}  , 'linewidth',1.6);
                    scatter(hyp1 , Dp, 40, hyp2_style(jj,:),'filled');
                end
                
            end
        end
        xlabel('R')
        ylabel(VOIstr{VOIid})
        pp = [pp,p];
        axis tight; box on
    end
    
    % Build a intelligible legend
    lleg = [];
    legenf = cell(1,Nhyper2+Nhyper3 );
    for jj =1:length(hyperparams{2,1})
        ll = plot(hyperparams{1,1}(1), Dp(jj), 'linewidth', 2,'color' , hyp2_style(jj,:)) ;
        lleg = [lleg,ll];
        legenf{1,jj} = ['\lambda_0=' , num2str(hyperparams{2,1}(jj))];
    end
    
    for kk =1:length(hyperparams{3,1})
        ll = plot(hyperparams{1,1}(1), Dp(kk), 'linewidth', 2,'color' , 'k','linestyle', hyp3_style{1,kk} ) ;
        lleg = [lleg,ll];
        legenf{1,length(hyperparams{2,1})+kk} = [' \alpha_0=' , num2str(hyperparams{3,1}(kk))];
    end
    
    h = legend(lleg,legenf);
    set(gcf, 'Position', [389         539        1124         295])
    linkaxes(pp)
    
    set(gcf, 'position', [486 1388 1678 423])
    %legend(modelstr)
    rect = [0.001, 0.25, .1, .25];
    set(h, 'Position', rect)
    
    
    %saveas(gcf,['~/Documents/Data/figures/20210211/','xval_',dataset_str,'_',cell2mat(kept_region),'_collapsed', num2str(collapse_conditions),VOIstr{1,VOIid},'.svg'])
    
end




%% Plot GCP-Decomposition Summary

% Select Model
hyp1 = 2;
hyp2 = 2;
hyp3 = 2;
hyp4 = 1;

% Reference Model
Refid = ref_tot(hyp1,hyp2,hyp3,hyp4);

% Selected Model
model_cur = gathered_cp{hyp1,hyp2,hyp3,hyp4};

% Neuron Groups
Pg = experimental_parameters.neuron_group;
corespid = experimental_parameters.neuron_idgrp;

% Plot regions/Layers
region_separators = 0.5+[0;find([diff(corespid(:,2));1])];
region_text = region_separators(1:end-1) + 0.5*(region_separators(2:end)-region_separators(1:end-1));

% Plot Colors
colortot = cell(1,4);

% Neurons
layrcolor = colorm(length(unique(corespid)));
layrcolor = layrcolor(corespid(:,1),:);
colortot{1,1} = layrcolor;

% Condition
if size(model_cur{3,1},1) ==3
    colortot{1,3} = [[0,0,1];[1,0,0];[0.5,0,0.5]];
    titestot = {'Vest';'Visu';'Both'};
    hand_reorder_id = [5,6,7,3,1,8,2,4]; % For standard
else
    colortot{1,3} = [[1,1,0];[0,1,1]; [0,0,1];[1,0,0];[0.5,0,0.5]];
    titestot = {'Oppo';'Same';'Vest';'Visu';'Both'};
    hand_reorder_id = [2,5,6,3,1,8,4,7];  % For mismatch
end




hand_reorder_id = 1:size(model_cur,2);
hand_reorder_id = [4,6,7,2,1,3,5];


% Trials
colortot{1,4} = 'r';

II = sum(used_dim);
JJ = size(model_cur,2);
ii_idx = find(used_dim);

% Data Labels
xlabelstot = cell(2,4);
xlabelstot{1,1} = region_text;
xlabelstot{2,1} = cellstr(dict_region');
xlabelstot{1,2} = 0:20:size(Xobs,2);
xlabelstot{2,2} = 0:20:size(Xobs,2);
xlabelstot{1,3} = 1:length(experimental_parameters.dict_cond);
xlabelstot{2,3} = titestot;
xlabelstot{1,4} = 1:size(Xobs,4)/2;
xlabelstot{2,4} = 1:size(Xobs,4)/2;

% CP-Factor Labels
titlestot = {'Loadings','Dynamics','Conditions','Trials'};

fastplot = 0;



figure
for ii=1:II
    for jj=1:JJ
        subplot(JJ,II, ii+(jj-1)*II); hold on
        
        % (ref) Factor
        CPij = model_cur{ii,hand_reorder_id(jj)};
        
        
        
        CPred = CPij(:,Refid);
        
        if ii==1
            % Neuron Loadings
            % Group Sum of Squared Loadings
            CPij = Pg{1,1}'*(CPij).^2;
            
            % Reference
            CPred = mean(CPij,2);
            
            % Normalized CP: = mean average Loading
            CPred_norm = (1./sum(Pg{1,1}))'.*CPred;
        end
        
        % Xaxis
        xx = 1:size(CPij,1);
        
        
        
        if ii==2
            % Temporal Loadings
            % Average +/- std
            mu = mean(CPij,2);
            mup = mu + std(CPij,[],2);
            mum = mu - std(CPij,[],2);
            plot(xx, mu,'color','r','linewidth',1.5)
            patch([xx(:); flipud(xx(:))], [mup(:); flipud(mum(:))], 'k', 'FaceAlpha',0.2)
            
            % Plot all
            %for nn=1:24;plot(CPij(:,nn));end
            
        elseif ii ==1
            % Neuron Loadind
            
            if fastplot
                % Box Plots
                boxplot(CPij','colors', 'k')
            else
                % Violin Plots
                CPtmp = CPij';
                IDtmp = repelem(1:size(CPij,1), size(CPij,2))';
                ppi = violinplot(CPtmp(:),IDtmp, 'ViolinAlpha', 1,'ShowData',false,'BoxWidth',0.1,'BoxColor', [0,0,0],'EdgeColor', [0 0 0]);
                ylimc = ylim;
                for vv = 1:size(CPij,1)
                    ppi(vv).ViolinColor = colortot{1,ii}(vv,:);
                    ppi(vv).MedianPlot.SizeData = 20;
                end
                
                % Plot region Separators
                for rsep = 1:length(region_separators)
                    line([region_separators(rsep) region_separators(rsep)],[ylimc(1) ylimc(2)],'linewidth',1.5,'color', 'k')
                end

                ylabel(['Component#', num2str(jj)])
            end
            
            % Normalized Neuron Loadings
            yyaxis right
            scatter(xx,CPred_norm,30,colortot{1,ii},'filled')
            scatter(xx,CPred_norm,30,'k','+')
            scatter(xx,CPred_norm,30,'k')
            ylabel('E (\oplus)')
            
            ax = gca;
            ax.YAxis(1).Color = 'k';
            ax.YAxis(2).Color = 'k';
            axis tight
            %ylim([0,0.008])
            yyaxis left
            %ylim([0,0.5])
            
            
        else
            % Task/Condition Loadings
            if fastplot
                % Box Plots
                boxplot(CPij','colors', 'k')
            else
                % Violin Plots
                CPtmp = CPij';
                IDtmp = titestot(repelem(1:size(CPij,1), size(CPij,2))');
                ppi = violinplot(CPtmp(:),IDtmp,'GroupOrder',titestot, 'ViolinAlpha', 1,'ShowData',false,'BoxWidth',0.1,'BoxColor', [0,0,0],'EdgeColor', [0 0 0]);
                for vv = 1:size(CPij,1)
                    ppi(vv).ViolinColor = colortot{1,ii}(vv,:);
                end
            end
            
            axis tight
            plot(xlim, 0*xlim, 'k')
            
            
        end
        
        set(gca,'xtick',xlabelstot{1,ii},'XTickLabel',xlabelstot{2,ii})
        box on;
        axis tight
        
        if jj==1
            title(titlestot {1,ii})
        end
    end
end


set(gcf,'position',[1601         390         884         921])
%print(['~/Documents/Data/figures/20210211/','xval_',dataset_str,'_',cell2mat(kept_region),'_collapsed', num2str(collapse_conditions), '_CP.svg'], '-dsvg','-painters')


%% Factor Correlations



% Build Correlation Matrix for all xval-folders
corr_tot = zeros(size(model_cur,1),size(model_cur{1,1},2),JJ,JJ);
for curfolder = 1:size(model_cur{1,1},2)
    for dimi=1:size(models_reordered2{1},2)
        
        % Get CP and normalize if necessary
        CP_i = models_reordered2{hyp1,hyp2,hyp3,hyp4,curfolder}{1,dimi};
        CP_i = CP_i(:,hand_reorder_id);
        CP_i = CP_i./sqrt(sum(CP_i.^2,1));
        
        % Absolute Correlation matrix
        corr_tot(dimi,curfolder,:,:) = abs(CP_i'*CP_i);
    end
end

corr_avg = zeros(size(model_cur,1),JJ,JJ );
for dimi=1:size(models_reordered2{1},2)
    corr_avg(dimi,:,:) = squeeze(mean(corr_tot(dimi,:,:,:),2));
end

% Average accross folder and plot
figure
for dimi=1:size(models_reordered2{1,1},2)
    subplot(1,size(models_reordered2{1,1},2),dimi)
    imagesc(squeeze(corr_avg(dimi,:,:)))
    colormap(hot); colorbar;
    ylabel('r_j'); xlabel('r_i'); 
    title(titlestot{dimi})
    caxis([0 1 ])
end
set(gcf,'position', [803        1592        1098         251])
%print(['~/Documents/Data/figures/20210211/','xval_',dataset_str,'_',cell2mat(kept_region),'_collapsed', num2str(collapse_conditions), '_correlations.svg'], '-dsvg','-painters')

%% Estimate to which extent a given factor modulates a (population of) neurons

% Average square neuron loadings for each component
dimoi = 1;
DDoi = size(models_reordered2{hyp1,hyp2,hyp3,hyp4,curfolder}{1,dimoi},1);
ddoi = size(models_reordered2{hyp1,hyp2,hyp3,hyp4,curfolder}{1,dimoi},2);
CP_i_pct = zeros(size(model_cur{1,1},2),DDoi,ddoi);
for curfolder = 1:size(model_cur{1,1},2)
    CP_i = models_reordered2{hyp1,hyp2,hyp3,hyp4,curfolder}{1,dimoi};
    
    CP_i = CP_i(:,hand_reorder_id).^2;
    CP_i_pct(curfolder,:,:) = CP_i;%./sum(CP_i,2);
end
CP_i_pct = squeeze(mean(CP_i_pct,1));


figure
discrim_str = {'region','layer','ctype','neurons'};
% Plot the results depending on how neurons are grouped
for discrim_id = 1:length(discrim_str)
    
    % Neuron group
    neuron_group_discriminant_cur = discrim_str{1,discrim_id};
    
    subplot(2,ceil(length(discrim_str)/2),discrim_id)
    if strcmp(neuron_group_discriminant_cur,'neurons')
        % Plot all neurons idividually
        indicator_ctype = build_neuron_group('ctype',record_ctype,record_layr, record_region,dict_ctype,dict_layr,dict_region);
        [indicator_layer, ~,neuron_strgp]  = build_neuron_group('layer',record_ctype,record_layr, record_region,dict_ctype,dict_layr,dict_region);
        
        % Reorder neurons
        id_ctype = sum(indicator_ctype.*(1:size(indicator_ctype,2)),2);
        id_layer = sum(indicator_layer.*(1:size(indicator_layer,2)),2);
        [~,new_neuron_order] = sort(id_ctype);
        
        % Gets proper layer id
        id_layer = find(diff([id_layer(new_neuron_order);100]));
        xx_layer = floor([0,id_layer(1:end-1)'] + 0.5*diff([0,id_layer']));
        xlegend = cellfun(@(x,y) [x,' | ',y], neuron_strgp(:,1),neuron_strgp(:,2), 'UniformOutput', false);
        
        % Reorder and Normalize neuron loadings
        var_cur = CP_i_pct(new_neuron_order,:)'; 
        var_cur = var_cur./sum(var_cur,1);
        
        % Plot
        imagesc(var_cur); 
        colorbar; axis tight; hold on;
        for separator = 1:length(id_layer)
            line([id_layer(separator),id_layer(separator)], ylim,'color','w','linewidth',2)
        end
        
    else
        % Use another type of discriminant
        [indicatorff,neuron_idgrp,neuron_strgp] = build_neuron_group(neuron_group_discriminant_cur,record_ctype,record_layr, record_region,dict_ctype,dict_layr,dict_region);
        var_cur = CP_i_pct'*indicatorff; var_cur = var_cur./sum(var_cur,1);
        
        
        % Get grid for text
        xgrid = repmat(1:size(var_cur,2),size(var_cur,1),1);
        ygrid = repmat(1:size(var_cur,1),size(var_cur,2),1)';
        tgrid = num2cell(round(var_cur,2)); % extact values into cells
        tgrid = cellfun(@num2str, tgrid, 'UniformOutput', false); % convert to string
        
        % Get Neuron Groups Id Tot
        if strcmp(neuron_group_discriminant_cur,'ctype')
            xlegend = cellfun(@(x,y,z) [x,' | ',y,' | ',z], neuron_strgp(:,1),neuron_strgp(:,2), neuron_strgp(:,3), 'UniformOutput', false);
        elseif strcmp(neuron_group_discriminant_cur,'layer')
            xlegend = cellfun(@(x,y) [x,' | ',y], neuron_strgp(:,1),neuron_strgp(:,2), 'UniformOutput', false);
        elseif strcmp(neuron_group_discriminant_cur,'region')
            xlegend = neuron_strgp;
        else
            xlegend = {};
        end
        
        % Plot
        imagesc(var_cur)
        hold on;axis tight
        text(xgrid(:), ygrid(:), tgrid, 'HorizontalAlignment', 'Center')
        xx_layer = 1:size(var_cur,2);
        colormap(hot); colorbar;
        
    
    end
    
    ylabel('CP #');xticks(xx_layer);xticklabels(xlegend);xtickangle(25)
    title(['% load on ' , discrim_str{discrim_id}])
    
end
set(gcf, 'position', [1601          79        1678         860])
%print(['~/Documents/Data/figures/20210211/','xval_',dataset_str,'_',cell2mat(kept_region),'_collapsed', num2str(collapse_conditions), '_contributions.svg'], '-dsvg','-painters')

%%








%% Look at Neuron Overlapp


dimoi = 1;
roi_1 = 1;
roi_2 = 7;
Dimoi = size(models_reordered2{hyp1,hyp2,hyp3,hyp4,curfolder}{1,dimoi},1);

corr_ijk = zeros(size(model_cur,1),Dimoi,Dimoi);
for curfolder = 1:size(model_cur{1,1},2)
    CP_ij = models_reordered2{hyp1,hyp2,hyp3,hyp4,curfolder}{1,dimoi};
    CP_ij = CP_ij(:,hand_reorder_id);
    CP_ij = CP_ij./sqrt(sum(CP_ij.^2,1));
    
    CP_1 = CP_ij(:,roi_1);
    CP_2 = CP_ij(:,roi_2);
    corr_ijk(curfolder,:,:) = CP_1*CP_2';

end
corr_ijk = Pg{1}'*diag(squeeze(mean(corr_ijk,1)));


figure
plot(corr_ijk)
title(titlestot{dimi})
%caxis([0 1 ])


%% Helper 

function [indicatorff,neuron_idgrp,neuron_strgp] = build_neuron_group(neuron_group_discriminant,record_ctype,record_layr, record_region,dict_ctype,dict_layr,dict_region)




if strcmp(neuron_group_discriminant,'ctype') || strcmp(neuron_group_discriminant,'neurons')
    record_ids = [record_ctype,record_layr, record_region];
    dict_tot = cell(1,size(record_ids,2));
    dict_tot{1,end-2} = dict_ctype;
    dict_tot{1,end-1} = dict_layr;
    dict_tot{1,end}   = dict_region;
elseif strcmp(neuron_group_discriminant,'layer')
    record_ids = [record_layr,record_region];
    dict_tot = cell(1,size(record_ids,2));
    dict_tot{1,end-1} = dict_layr;
    dict_tot{1,end}   = dict_region;
elseif strcmp(neuron_group_discriminant,'region')
    record_ids = [record_region];
    dict_tot = cell(1,size(record_ids,2));
    dict_tot{1,end}   = dict_region;
    
end

% Reorder and Sum-up neurons
[indicatorff,neuron_idgrp,neuron_strgp] = get_indicator(record_ids,dict_tot);

end
