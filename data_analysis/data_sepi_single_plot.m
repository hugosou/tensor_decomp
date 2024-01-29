
%% Load data
%load('/home/sou/Documents/Data/dresults/singlefit_standard_stitching1_flipback0_collapsed1_RSPdRSPgSC_2021_03_03_06_25.mat')
%load('/home/sou/Documents/Data/dresults/single_fit_standard_RSPdRSPgSCSUBV1_collapsed1_2021_02_12_18_53.mat')
load('/home/sou/Documents/Data/dresults/singlefit_standard_stitching1_flipback0_collapsed1_RSPdRSPgSCSUBV1_2021_08_03_04_10.mat')
dataset_str = 'standard';
%%
addpath(genpath('~/Documents/MATLAB/tensor_decomp/'))
% Extract experimental parameters
param_names = fieldnames(experimental_parameters);
for parami = 1:length(param_names)
    eval([param_names{parami} '=experimental_parameters.' param_names{parami},';']);
end
dataset_str = 'standard';

%% Get similarities
used_dim = [1,1,1,1,1];
[smlty_tot,ref_tot,permt_tot,~,sig_tot] = get_similarity_tot(models_tot,used_dim);
kept_region = cellstr((dict_region(:)')); kept_region = [kept_region{:}];

%% Non Collapsed
models_squeezed = squeeze_model(models_tot, observed_data, [1,0,0,1,0]);
[smlty_tot,ref_tot,permt_tot,~,sig_tot]   = get_similarity_tot(models_squeezed,[1,1,1,1]);

%% Collapsed
Xdims = size(observed_data);
[smlty_tot,ref_tot,permt_tot,~,sig_tot]   = get_similarity_tot(models_tot,[1,1,1],...
1:size(models_tot,5),[1,0,1], reshape(observed_data,[Xdims(1:2),prod(Xdims(3:end))]));


%% Colapsed No Missing
[smlty_tot,ref_tot,permt_tot,~,sig_tot]   = get_similarity_tot(models_tot,[1,1,1]);
models_squeezed = models_tot;
%%

models_reordered= reorder_cps(models_squeezed,permt_tot,sig_tot);
gathered_cp      = gather_cps(models_reordered);


mdims =  size(models_squeezed);
smlty_tot = reshape(smlty_tot, [mdims(1:end-1),mdims(end)-1]);
ref_tot   = reshape(ref_tot  , [mdims(1:end-1)]);
%%

save_figure = 0;


%% Plot similarities accross initialization
Nhyp1 = size(models_tot,1);
Nhyp2 = size(models_tot,2);
Nhyp3 = size(models_tot,3);
Nhyp4 = size(models_tot,4);

hyp2_style = [linspace(0,1,Nhyp2).^2',0.5*linspace(1,0,Nhyp2).^2',linspace(1,0,Nhyp2)'];
hyp3_style = {'-','-.','-.-',':'};
legenf = cell(1,Nhyp2*Nhyp3);

figure
for nhyp2=1:Nhyp2
    for nhyp3=1:Nhyp3
        for nhyp4=1:Nhyp4
            
            sim_cur = squeeze(smlty_tot(:,nhyp2,nhyp3,nhyp4,:));

            mu_cur = mean(sim_cur,2);
            si_cur = std(sim_cur,[],2);
            
            subplot(1,Nhyp4,nhyp4); hold on
            dd=errorbar(hyperparams{1} , mu_cur,si_cur, 'color', hyp2_style(nhyp2,:), 'linewidth',1.6, 'linestyle',hyp3_style{nhyp3});
            d.Bar.LineStyle = 'dotted';
            
            ylim([0 1])
            box on
            xlabel('R')
            ylabel('Similarities')
            
            legenf{1,nhyp3+(nhyp2-1)*Nhyp3} = ['\lambda_0=',num2str(hyperparams{2,1}(nhyp2)), ' \alpha_0=',num2str(hyperparams{3,1}(nhyp3))];
            title(hyperparams{4}(nhyp4))
        end
    end
end
legend(legenf)
%legend({'','','',''})
set(gcf,'position', [ 461   170   633   179])

%% Plot similarities accross initialization
Nhyp1 = size(models_tot,1);
Nhyp2 = size(models_tot,2);
Nhyp3 = size(models_tot,3);
Nhyp4 = size(models_tot,4);

hyp2_style = [linspace(0,1,Nhyp2).^2',0.5*linspace(1,0,Nhyp2).^2',linspace(1,0,Nhyp2)'];
hyp3_style = {'-','-.','-.-',':'};
legenf = cell(1,Nhyp2*Nhyp3);

figure
for nhyp2=1:Nhyp2
    for nhyp3=1:Nhyp3
        for nhyp4=1:Nhyp4
            
            sim_cur = squeeze(smlty_tot(nhyp2,:,nhyp3,nhyp4,:));

            mu_cur = mean(sim_cur,2);
            si_cur = std(sim_cur,[],2);
            
            subplot(1,Nhyp4,nhyp4); hold on
            dd=errorbar(hyperparams{1} , mu_cur,si_cur, 'color', hyp2_style(nhyp2,:), 'linewidth',1.6, 'linestyle',hyp3_style{nhyp3});
            d.Bar.LineStyle = 'dotted';
            
            ylim([0 1])
            box on
            xlabel('R')
            ylabel('Similarities')
            
            legenf{1,nhyp3+(nhyp2-1)*Nhyp3} = ['\lambda_0=',num2str(hyperparams{2,1}(nhyp3)), ' \alpha_0=',num2str(hyperparams{3,1}(nhyp4))];
            title(hyperparams{4}(nhyp4))
        end
    end
end
legend(legenf)
%legend({'','','',''})
set(gcf,'position', [ 461   170   633   179])



%% Plot 1 model
addpath('./Violinplot')
% Use ref or avg
use_ref = 1;
use_avg = 0;

% Select Model
hyp1 = 1;
hyp2 = 2;
hyp3 = 1;
hyp4 = 1;

% Reference Model
Refid = ref_tot(hyp1,hyp2,hyp3,hyp4);


% Selected Model is reference or averaged model across init
if use_ref
    model_cur  = models_reordered{hyp1,hyp2,hyp3,hyp4,Refid};
else
    models_cur = gathered_cp{hyp1,hyp2,hyp3};
    model_avg  = cell(1,size(models_cur,1));
    for dimi=1:size(models_cur,1)
        model_avg{1,dimi} = cell2mat(cellfun(@(x) mean(x,2),models_cur(dimi,:), 'UniformOutput',false));
    end
end


% Align factor signs
model_cur = signorder_cp(model_cur,2);


%plot_cp(models_tot{hyp1,hyp2,hyp3,hyp4,Refid})


% Neuron Groups
Pg = experimental_parameters.neuron_group;

corespid = experimental_parameters.neuron_idgrp;
% Plot regions/Layers
region_separators = 0.5+[0;find([diff(corespid(:,2));1])];
region_text = region_separators(1:end-1) + 0.5*(region_separators(2:end)-region_separators(1:end-1));

% Plot Colors
colortot = cell(1,4);

% Neurons
cellcolor = colorm(length(unique(corespid)));
cellcolor = cellcolor(corespid(:,1),:);
colortot{1,1} = cellcolor;

%UNCOMMENT FOR  EARLIER VERSION
% if size(model_cur{1,3},1) ==3
%     colortot{1,3} = [[0,0,1];[0.5,0,0.5];[1,0,0]];
%     titestot = {'Vest';'Both';'Visu'};
%     hand_reorder_id = [4,7,6,2,1 ,5,3];
% else
%     colortot{1,3} = [[1,1,0];[0,1,1]; [0,0,1];[1,0,0];[0.5,0,0.5]];
%     titestot = {'Oppo';'Same';'Vest';'Visu';'Both'};
%     hand_reorder_id = [2,6,5,3,1,4,7];  % For mismatch
% end
% colortot{1,4} =repmat([0,0,0],size(model_cur{1,4},1),1);
% hand_reorder_id = 1:size(model_cur{1},2);
% hand_reorder_id = [3,7,6,2,1,4,5];
% 


colortot{1,3} = [[0,0,1];[0.5,0,0.5];[1,0,0]];
titestot = {'Vest';'Both';'Visu'};
hand_reorder_id = [3,7,6,2,1,4,5];
%hand_reorder_id = 1:size(model_cur{1},2);

II = size(model_cur,2);
JJ = size(model_cur{1},2);



% Data Labels
xlabelstot = cell(2,4);
xlabelstot{1,1} = region_text;
xlabelstot{2,1} = cellstr(dict_region');
xlabelstot{1,2} = 0:20:size(model_cur{2},1);
xlabelstot{2,2} = 0:20:size(model_cur{2},1);
xlabelstot{1,3} = 1:length(titestot);
%xlabelstot{1,3} = 1:length(experimental_parameters.condition);
xlabelstot{2,3} = titestot;
xlabelstot{1,4} = size(model_cur{3},1)/20;
xlabelstot{2,4} = size(model_cur{3},1)/20;

% CP-Factor Labels
titlestot = {'Loadings','Dynamics','Conditions','Trials'};

fastplot= 0;
use_avg = 0;


condition_design = repmat(eye(3), [1,1,10]);
%condition_design = permute(condition_design,[1,3,2]);
condition_design = tensor_unfold(condition_design,1);

figure
for ii=1:II
    for jj=1:JJ
        subplot(JJ,II, ii+(jj-1)*II); hold on
        
        % (ref) Factor
        CPij = model_cur{1,ii}(:,hand_reorder_id(jj));

        
        if ii==1
            % Neuron Loadings
            % Group Sum of Squared Loadings
            CPij = Pg{1,1}'*(CPij).^2;
            
            % Normalized CP: = mean average Loading
            CPred_norm = (1./sum(Pg{1,1}))'.*CPij;
        end
        
        % Xaxis
        xx = 1:size(CPij,1);
        
        if ii==2
            % Temporal Loadings
            plot(xx, CPij,'color','k','linewidth',1.5)
            
        elseif ii ==1
            % Neuron Loadind
            
            scatter(xx,CPij,60, colortot{1,ii}, 'filled')
            scatter(xx,CPij,60, 'k')
            
            % Normalized Neuron Loadings
            if use_avg
                yyaxis right
                scatter(xx,CPred_norm,60,colortot{1,ii},'+','linewidth',2)
                scatter(xx,CPred_norm,60,'k','linewidth',1)
                ylabel('E (\oplus)')
                ax = gca;ax.YAxis(1).Color = 'k';ax.YAxis(2).Color = 'k';
                axis tight
                yyaxis left
            end
            
            ylimc = ylim;
            % Plot region Separators
            for rsep = 1:length(region_separators)
                line([region_separators(rsep) region_separators(rsep)],[ylimc(1) ylimc(2)],'linewidth',1.5,'color', 'k')
            end
            ylabel(['Component#', num2str(jj)])
            

        else
            %scatter(xx, CPij, 50,colortot{1,ii},'filled')
         
            % Task/Condition Loadings
            if fastplot
                % Box Plots
                scatter(xx, CPij)
            else
                % Violin Plots
                cnd_id = sum(condition_design.*...
                    (1:unique(size(condition_design,1)))')';
                CPtmp = CPij';
                %IDtmp = titestot(repelem(1:size(CPij,1), size(CPij,2))');
                %ppi = violinplot(cnd_id(:),cnd_id,'GroupOrder',titestot, 'ViolinAlpha', 1,'ShowData',false,'BoxWidth',0.1,'BoxColor', [0,0,0],'EdgeColor', [0 0 0]);
                ppi = violinplot(CPij,cnd_id ,'ViolinAlpha', 1,'ShowData',true,'BoxWidth',0.1,'BoxColor', [0,0,0],'EdgeColor', [0 0 0]);
                for vv = 1:length(unique(cnd_id))
                    ppi(vv).ViolinPlot.FaceAlpha = 0.1;
                    ppi(vv).ScatterPlot.MarkerEdgeColor = 'k';
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


if save_figure 
    print(['~/Documents/Data/figures/20210318/','single_',dataset_str,'_',kept_region,'_collapsed', num2str(collapse_conditions), '_CP.svg'], '-dsvg','-painters')
end




%% Correlation Matrix

% Build Correlation Matrix for all xval-folders
corr_cur = zeros(size(models_reordered{1},2),JJ,JJ);

for dimi=1:size(models_reordered{1},2)
    % Get CP and normalize if necessary
    CP_neuron_reordered = model_cur{1,dimi};
    CP_neuron_reordered = CP_neuron_reordered(:,hand_reorder_id);
    CP_neuron_reordered = CP_neuron_reordered./sqrt(sum(CP_neuron_reordered.^2,1));
    
    % Absolute Correlation matrix
    corr_cur(dimi,:,:) = (abs(CP_neuron_reordered)'*abs(CP_neuron_reordered));
end


% Average accross folder and plot
figure
for dimi=1:size(models_reordered{1},2)
    subplot(1,size(models_reordered{1},2),dimi)
    imagesc(squeeze(corr_cur(dimi,:,:)))
    colormap(hot); colorbar;
    ylabel('r_j'); xlabel('r_i');
    title(titlestot{dimi})
    %caxis([0 1 ])
end


set(gcf,'position', [803        1592        1098         251])


if save_figure 
    print(['~/Documents/Data/figures/20210318/','single_',dataset_str,'_',kept_region,'_collapsed', num2str(collapse_conditions), '_correlations.svg'], '-dsvg','-painters')
end


%% Estimate to which extent a given factor modulates a (population of) neurons

% Average neuron loadings for each component
dimoi = 1;
%CP_i_pct = model_cur{1,dimoi}.^2;
CP_i_pct = abs(model_cur{1,dimoi});
CP_i_pct = CP_i_pct(:,hand_reorder_id);


figure
discrim_str = {'region','layer','ctype','neurons'};
% Plot the results depending on how neurons are grouped
for discrim_id = 1:length(discrim_str)
    
    % Neuron group
    neuron_group_discriminant_cur = discrim_str{1,discrim_id};
    
    subplot(2,ceil(length(discrim_str)/2),discrim_id)
    if strcmp(neuron_group_discriminant_cur,'neurons')
        % Plot all neurons idividually
        indicator_ctype = build_neuron_group('ctype',record_celltype,record_layer, record_region,dict_ctype,dict_layr,dict_region);
        [indicator_layer, ~,neuron_strgp]  = build_neuron_group('layer',record_celltype,record_layer, record_region,dict_ctype,dict_layr,dict_region);
        
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
        [indicatorff,neuron_idgrp,neuron_strgp] = build_neuron_group(neuron_group_discriminant_cur,record_celltype,record_layer, record_region,dict_ctype,dict_layr,dict_region);
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


if save_figure 
    print(['~/Documents/Data/figures/20210318/','single_',dataset_str,'_',kept_region,'_collapsed', num2str(collapse_conditions), '_contributions.svg'], '-dsvg','-painters')
end



%% Break down Colinear Factors by neurons

% Plot Parameters
R1 = 4;
R2 = 5;
use_pct = 1;
fold_scatter = 0;
neuron_group_discriminant_cur = 'layer';

% Threshold loadings
thr_loadings_tot = 0*0.05;

% Partition Space
thr_loadings_pct = 0*0.2;


% Sort Neurons by group
[indicator_cur, neuron_idgrp,neuron_strgp]  = build_neuron_group(neuron_group_discriminant_cur,record_celltype,record_layer, record_region,dict_ctype,dict_layr,dict_region);

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

% Gather CPs of interests
CP_1 = model_cur{1,1}(:,hand_reorder_id([R1,R2])); 
CP_2 = model_cur{1,2}(:,hand_reorder_id([R1,R2])); 
CP_3 = model_cur{1,3}(:,hand_reorder_id([R1,R2])); 

% Threshold CP with small loadings
thred_tot = max(abs(CP_1),[],2) > thr_loadings_tot;

% Threshold CP with small loadings in % for R1 and R2
CP_1 = CP_1./sum(abs(model_cur{1,1}),2);
thred_pct = sqrt(sum(CP_1.^2,2)) > thr_loadings_pct;

% Get thresholded Loadings
kept_neurons = find(thred_tot.*thred_pct);
kept_indicat = indicator_cur(kept_neurons,:);
CP_1 = CP_1(kept_neurons,:);
CP_R12 = cell(1,3);
CP_R12{1,1} = CP_1;
CP_R12{1,2} = CP_2;
%CP_R12{1,3} = experimental_parameters.condition_design*CP_3./sum(experimental_parameters.condition_design,2);
CP_R12{1,3} = CP_3;

% Polar to cartesian and vice versa
polar_coord = @(X) [sqrt(X(:,1).^2+X(:,2).^2), atan2(X(:,2),X(:,1))];
carte_coord = @(X) [X(:,1).*cos(X(:,2)),X(:,1).*sin(X(:,2))];

% Partition space using angle
partition_N = 8;
partition_boundaries = linspace(-pi/partition_N, 2*pi-pi/partition_N,partition_N+1 );
partition_middles    = 0.5 * (partition_boundaries(1:(end-1))+partition_boundaries(2:end));
partition_middles(partition_N+1) = 2*pi;

% Angle for each factors
CP1_theta = polar_coord(CP_1); 
CP1_theta = CP1_theta(:,2);
CP1_theta = CP1_theta.*(CP1_theta>=0) + (CP1_theta+2*pi).*(CP1_theta<0);

% Fold loadings if sign doesn't matter
if fold_scatter
    CP1_theta = mod(CP1_theta,pi);
end

% Assign loadings to a given group
[~, partition_idx] = min(abs(CP1_theta-partition_middles),[],2);
partition_idx(partition_idx==partition_N+1) = 1;

% Fold loadings if sign doesn't matter
if fold_scatter
    partition_idx(partition_idx == partition_N/2+1) = 1;
end

% Build "centroids"
CPthy = cell(1,3);
CPthy{1,1} = [[+1,0];[1,1];[0,1];[-1,1];[-1,0];[-1,-1];[0,-1];[1,-1]];
CPthy{1,2} = CP_2;
%CPthy{1,3} = experimental_parameters.condition_design*CP_3./sum(experimental_parameters.condition_design,2);
CPthy{1,3} = CP_3;

Xdims = size(observed_tensor);
Xthy = tensor_reconstruct(CPthy);
Xthy = mean(reshape(Xthy, [8, Xdims(2:end)]),4);
Xthy = tensor_unfold(Xthy,1);

% Layer/Regions/ctype colors
corespid = neuron_idgrp;
cellcolor = colorm(length(unique(corespid)));
cellcolor = cellcolor(corespid(:,1),:);



% Reorder plottings 
if fold_scatter
    barplots_loc  = [12,6,5,4];
    centroid_loc  = [26,8,5,2];
else
    barplots_loc  = [12,6,5,4,10,16,17,18];
    centroid_loc  = [26,8,5,2,20,38,41,44];
end
ppct = [];

figure; 
% Plot Centroids and contributions
for ncond_i = 1:length(barplots_loc)
    
    % Neurons in current partition
    neuron_partition_i     = find(partition_idx==ncond_i);
    
    % Number of neurons in current partition
    neuron_partition_group = sum(kept_indicat(neuron_partition_i,:),1);
    
    % Use percentages or cell count 
    if use_pct
        neuron_partition_group = neuron_partition_group./sum(kept_indicat,1);
    end
    
    % Plot "centroids"
    subplot(6,9,centroid_loc(ncond_i)); hold on
    for ii = 1:size(colortot{1,3},1)
        Tcur = (1:(size(Xthy,2)/size(colortot{1,3},1))) + (ii-1) * size(Xthy,2)/size(colortot{1,3},1);
        plot(Tcur, Xthy(ncond_i,Tcur),'color',colortot{1,3}(ii,:))
    end
    axis off; box on; axis tight
    if barplots_loc(ncond_i) == 1;legend(titestot); end
 
    % Bar plots: # neurons in current region
    ppcti = subplot(6,3,barplots_loc(ncond_i)); hold on
    for ii = 1:size(neuron_partition_group,2)
       bar(ii,neuron_partition_group(ii), 'facecolor', cellcolor(ii,:))
    end
    
    if (barplots_loc(ncond_i)> 12) || (fold_scatter && (barplots_loc(ncond_i)== 12))
        xticks([1:size(neuron_partition_group,2)])
        set(gca,'xticklabel',xlegend,'fontsize',9); xtickangle(60)
    else
        set(gca,'xticklabel','')
    end
    axis tight; %box on
    ppct = [ppct,ppcti];
    
end
linkaxes(ppct)
if use_pct ;ylim([0 1]); end

% Plot Loadings + Regions
subplot(3,3,5); hold on

% Plot boundaries
rmax = polar_coord(CP_1); rmax = max(rmax(:,1),[],1);
partition_coundaries_cart = carte_coord([rmax*ones(partition_N,1),partition_boundaries(1:end-1)']);
for current_boundary = 1:size(partition_coundaries_cart,1)
    line([0,partition_coundaries_cart(current_boundary,1)], [0,partition_coundaries_cart(current_boundary,2)], 'color', 'k')
end
fimplicit(@(x,y) (x).^2 + (y).^2 -thr_loadings_pct^2,'color','k')

% Scatter loadings for each group
for ngroup_i = 1:size(kept_indicat,2)
    neuron_i = find(kept_indicat(:,ngroup_i));
    scatter(CP_1(neuron_i,1),CP_1(neuron_i,2),20, cellcolor(ngroup_i,:), 'filled')
    scatter(CP_1(neuron_i,1),CP_1(neuron_i,2),20, 'k')
end


box on; title('Neuron Loadings')
xlim([-rmax , rmax]); ylim([-rmax , rmax])
xlabel(['Factor #', num2str(R1)]) ;ylabel(['Factor #', num2str(R2)]); axis equal
set(gcf,'position', [1601          56        1215         931])



if save_figure 
    print(['~/Documents/Data/figures/20210318/','factor_', num2str(R1), 'VS', num2str(R2),'_pct', num2str(use_pct),'_',dataset_str,'_',kept_region,'_collapsed', num2str(collapse_conditions), '.svg'], '-dsvg','-painters')
end


%%











%%


model_cur  = models_reordered{hyp1,hyp2,hyp3,hyp4,Refid};
offset_cur = offset_tot{hyp1,hyp2,hyp3,hyp4,Refid};
decomp_cur = tensor_reconstruct(model_cur);

Xobs = observed_tensor;


Xhat = reshape(exp(decomp_cur+offset_cur),size(Xobs));
%Xhat = reshape(exp(decomp_cur+offset_cur),[901,70,10,3]);
%Xhat = permute(Xhat,[1,2,4,3]);

TT = size(Xobs,2);




for nn = 1:5
    
    Xobsn_tmp = Xobs(kept_neurons(nn), :, :,:);
    Xobsn = squeeze(mean(Xobsn_tmp, 4));
    Xobsv = squeeze(std(Xobsn_tmp, [], 4));
    
    Xhatn_tmp = Xhat(kept_neurons(nn), :, :,:);
    Xhatn = squeeze(mean(Xhatn_tmp, 4));
    Xhatv = squeeze(std(Xhatn_tmp, [], 4));
    
    figure; hold on
    for condi = 1:size(Xobsn,2)
        
        ta = ((condi-1)*TT+1):condi*TT;
        xo =  Xobsn(:,condi);
        xh =  Xhatn(:,condi);
        
        yyaxis left
        plot(ta,xo, 'color', colortot{1,3}(condi,:), 'linestyle', '-')
        
        yyaxis right
        plot(ta, Xhatn(:,condi), 'color', colortot{1,3}(condi,:), 'linestyle', '-.')
        
        regionc = cellstr(dict_region(neuron_idgrp(find(kept_indicat(nn,:)),2)));
        layerc  = cellstr(dict_layr(neuron_idgrp(find(kept_indicat(nn,:)),1)));
        regionlayer = [regionc{1}, '-', layerc{1} ];
        title(regionlayer)
        
    end
end

%%

Xobs_t = tensor_unfold(Xobs,4)';
Xhat_t = tensor_unfold(Xhat,4)';

Ntrial = size(Xobs_t,2);


Prbar = std(mean(Xobs_t,2));
Pbarr = mean(var(Xobs_t,[],1));
Pmu = (Ntrial*Prbar-Pbarr)/(Ntrial-1);


Pr = var(Xobs_t,[],1);
Prmr = var(Xobs_t-Xhat_t,[],1);

Pnoise =  Pbarr - Pmu;

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

