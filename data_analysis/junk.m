% Add master folders
%addpath(genpath('~/Documents/MATLAB/tensor_decomp2/'))

addpath(genpath('~/Documents/MATLAB/'))


data_folder = '/nfs/gatsbystor/hugos/data_sepi_all/';
resu_folder = '/nfs/gatsbystor/hugos/';

data_folder = '~/Documents/Data/data_sepi_all/';
resu_folder = '~/Documents/Data/data_sepi_all/';

% Dataset Parameters
flip_back_data = 0;
stitch_data    = 0;

% Which dataset to use: 'mismatch', 'standard'
type_data = 'standard';

% Load
name_data = [data_folder,'data_sepi_',type_data, '_stitching', num2str(stitch_data), '_flipback', num2str(flip_back_data)];
load(name_data)

% Keep regions from : 'N/A','RSPd','RSPg','SC','SUB','V1', 'Hip'
kept_region = {'RSPd','RSPg','SC','SUB','V1'};

% Attribute to group neurons: 'ctype','layer','region'
neuron_group_discriminant = 'layer';

% Load
data_sepi = data_sepi_load(data_sepi,kept_region,neuron_group_discriminant);

% Extract experimental parameters
param_names = fieldnames(data_sepi);
for parami = 1:length(param_names)
    eval([param_names{parami} '=data_sepi.' param_names{parami},';']);
end

Xobs_original = observed_tensor;

%%

% Xobs_original
% data_sepi.direction

TT = size(Xobs_original,2);


T = floor(TT/2);
ta1 = [1:T, fliplr(1:T)];
ta2 = [fliplr(1:T), 1:T];


Xobs_psth_1 = zeros(size(Xobs_original,1:3));
Xobs_psth_2 = zeros(size(Xobs_original,1:3));


XXobs_psth_1 = zeros(size(Xobs_original,1),size(Xobs_original,2)/2,size(Xobs_original,3));
XXobs_psth_2 = zeros(size(Xobs_original,1),size(Xobs_original,2)/2,size(Xobs_original,3));

for nn = 1:size(Xobs_original,1)
    
    % Only one session with multiple trial per neuron
    session_cur = find(data_sepi.observed_data(nn,1,1,:,1));
    direction_cur = data_sepi.direction(:,session_cur,:);
    
    for condition_cur = 1:size(Xobs_original,3)
        
        X1 = Xobs_original(nn,:,condition_cur,session_cur, find(data_sepi.direction(condition_cur,session_cur,:)==+1));
        X2 = Xobs_original(nn,:,condition_cur,session_cur, find(data_sepi.direction(condition_cur,session_cur,:)==-1));
        
        XX1 = X1(1,1:T,1,1,:)+fliplr(X1(1,T+(1:T),1,1,:));
        XX2 = fliplr(X2(1,1:T,1,1,:))+X2(1,T+(1:T),1,1,:);
        
        XXobs_psth_1(nn,:,condition_cur) = squeeze(mean(XX1,5));
        XXobs_psth_2(nn,:,condition_cur) = squeeze(mean(XX2,5));
        
        X1 = squeeze(mean(X1,5));
        X2 = squeeze(mean(X2,5));
        
        Xobs_psth_1(nn,:,condition_cur) = X1(:);
        Xobs_psth_2(nn,:,condition_cur) = X2(:);
        
        
        
        
    end
end

%%

%%

clc
CP1 = model_cur{1,1}(:,hand_reorder_id);
%CP1 = CP1./sum(abs(CP1),2);

comp = 6;
CP1comp = CP1(:,comp);


CP1comp = abs(CP1(:,1)).^2 .* abs(CP1(:,7)).^2;
[pctmod, neuron_index] = sort(abs(CP1comp), 'descend');

pctmod(1:5)'
neuron_index(1:5)'




%%
vline= (0:6)*35;

NN = [203];
save_dict =  '~/Documents/Data/figures/20211028/';
save_fig = 0;

% Fitted tensor
model_cur  = models_reordered{hyp1,hyp2,hyp3,hyp4,Refid};
offset_cur = offset_tot{hyp1,hyp2,hyp3,hyp4,Refid};
decomp_cur = tensor_reconstruct(model_cur);
Xhat = reshape(exp(decomp_cur+offset_cur),[size(observed_tensor,1:3),size(observed_tensor,5)]);



for nn = NN
    
    % PSTH ordered by lab frame position
    Xobsn1 = XXobs_psth_1(kept_neurons(nn), :,:);
    Xobsn2 = XXobs_psth_2(kept_neurons(nn), :,:);
    
    % PSTH 
    Xpsth1 = Xobs_psth_1(kept_neurons(nn), :,:);
    Xpsth2 = Xobs_psth_2(kept_neurons(nn), :,:);
    Xpsth = mean(sum(Xobs_original(nn,:,:,:,:),4),5);
    
    
    figure; hold on
    legend1 =[]; 
    legend2 =[];
    
    for condi = 1:size(Xobsn1,3)
        
        
        % Plot the lab frame position PSTH
        p = subplot(size(Xobsn1,3),6,5+(condi-1)*6); hold on
        ta = (1:T)/T*pi;
        plot(ta,Xobsn1(1,:,condi), 'color',  colortot{1,3}(condi,:)*0.3, 'linewidth', 1.5)
        plot(ta,Xobsn2(1,:,condi), 'color',  colortot{1,3}(condi,:)*0.7+0.3, 'linewidth', 1.5)
        if condi==size(Xobsn1,3)
            xlabel('Position (lab frame)')
        elseif condi==1
            legend('CCW','CW')
        end
        legend1 =[legend1,p]; box on;
        xticks([ 0 pi/2 pi])
        xticklabels({'0','\pi/2','\pi'})
        
        
        % Plot relative PSTH and fit
        q = subplot(size(Xobsn1,3),3,1+(condi-1)*3); hold on
        Xhatc = mean(Xhat(nn,:,condi,:),4);
        Xpsth = mean(sum(Xobs_original(nn,:,:,:,:),4),5);
        ta = ((1:TT)-1)/(TT-1)*2*pi;
        plot(ta,Xpsth(1,:,condi), 'color', colortot{1,3}(condi,:), 'linestyle', '-')
        plot(ta,Xhatc, 'color', colortot{1,3}(condi,:), 'linewidth', 1.5)
        ycur = ylim;
        line([pi,pi], [min(Xpsth(:)),max(Xpsth(:))], 'color','k');
        legend2 =[legend2,q];
        box on; 
        xlim([ta(1),ta(end)])
        ylim([min(Xpsth(:)),max(Xpsth(:))])
        xticks([ 0 pi 2*pi])
        xticklabels({'0','\pi','0'})
        ylabel( condition{condi})
        regionc = cellstr(dict_region(neuron_idgrp(find(kept_indicat(nn,:)),2)));
        layerc  = cellstr(dict_layr(neuron_idgrp(find(kept_indicat(nn,:)),1)));
        regionlayer = ['Neuron ' , num2str(kept_neurons(nn))];  
        if condi == 1 
            title(regionlayer)
            legend('PSTH','Fit')
        elseif condi==size(Xobsn1,3)
            xlabel('Position (relative)')
        end
        
        % Plot PSTH
        r = subplot(size(Xobsn1,3),3,2+(condi-1)*3); hold on
        plot(ta,Xpsth1(1,:,condi), 'color', colortot{1,3}(condi,:)*0.3, 'linewidth', 1.5)
        plot(ta,Xpsth2(1,:,condi), 'color', colortot{1,3}(condi,:)*0.6+0.4, 'linewidth', 1.5) 
        xlim([ta(1),ta(end)])
        ylim([min(min(Xpsth1(:)),min(Xpsth2(:))),max(max(Xpsth1(:)),max(Xpsth2(:)))])
        xticks([ 0 pi 2*pi])
        xticklabels({'0','\pi','0'})
        line([pi,pi], [min(min(Xpsth1(:)),min(Xpsth2(:))),max(max(Xpsth1(:)),max(Xpsth2(:)))], 'color','k');
        box on
        
        if condi == 1
            title([' (',regionc{1}, ' - ', layerc{1},')'])
            legend('CCW','CW')
        elseif condi==size(Xobsn1,3)
            xlabel('Position (relative)')
        end
        
        
    end
    
    

    linkaxes(legend1)
    linkaxes(legend2)
    set(gcf,'position', [2366         436         807         553])
    
    
    if save_fig
        print([save_dict, '/n', num2str(nn), '_psth.svg'], '-dsvg','-painters')
    end
end

%%

colort = [[0 0 1];[0.5 0 0.5];[1 0 0]]; 
for nn = NN
    figure;
    % Only one session with multiple trial per neuron
    session_cur = find(data_sepi.observed_data(nn,1,1,:,1));
    direction_cur = data_sepi.direction(:,session_cur,:);
    
    Xcur = Xobs_original(nn,:,:,session_cur,:);
    rmax = 0.4*max(Xcur(:));
    
    for condition_cur = 1:size(Xobs_original,3)
        
        MM = max(Xobs_original(:));
        %MM = 10
        
        X10 = Xcur(1,:,condition_cur,1,find(data_sepi.direction(condition_cur,session_cur,:)==+1));
        X20 = Xcur(1,:,condition_cur,1,find(data_sepi.direction(condition_cur,session_cur,:)==-1));
        %X10 = Xobs_original(nn,:,condition_cur,session_cur, find(data_sepi.direction(condition_cur,session_cur,:)==+1));
        %X20 = Xobs_original(nn,:,condition_cur,session_cur, find(data_sepi.direction(condition_cur,session_cur,:)==-1));
        
        
        
        subplot(size(Xobs_original,3),2,1+(condition_cur-1)*2)
        circle_raster(squeeze(X10), 'CCW', colort(condition_cur,:), rmax)
        if condition_cur==1
            title(['CCW: Neuron ' , num2str(kept_neurons(nn))])
        end
        
        
        subplot(size(Xobs_original,3),2,2+(condition_cur-1)*2)
        circle_raster(flipud(squeeze(X20)), 'CW', colort(condition_cur,:), rmax)
        if condition_cur==1
            title(['CW: Neuron ' , num2str(kept_neurons(nn))])
        end
        
        
    end
    
    set(gcf, 'Position', [1921         111         799         868])
    if save_fig
        print([save_dict, '/n', num2str(nn), '_raster.svg'], '-dsvg','-painters')
    end
end



%%
ta = (1:70)/10;


for nn = NN
    
    % Only one session with multiple trial per neuron
    session_cur = find(data_sepi.observed_data(nn,1,1,:,1));
    direction_cur = data_sepi.direction(:,session_cur,:);
    
    
    figure
    for condition_cur = 1:size(Xobs_original,3)
        
        X1 = Xobs_original(nn,:,condition_cur,session_cur, find(data_sepi.direction(condition_cur,session_cur,:)==+1));
        X2 = Xobs_original(nn,:,condition_cur,session_cur, find(data_sepi.direction(condition_cur,session_cur,:)==-1));

        X1 = squeeze(X1);
        X2 = squeeze(X2);
        
        subplot(size(Xobs_original,3), 2, (condition_cur-1)*2+1); hold on
        for kk = 1:size(X1,2)
            plot(ta, X1(:,kk), 'color', colort(condition_cur,:)*(kk-1)/size(X1,2), 'linewidth', 1.5 )
        end
        
        ylim0 = ylim;
        line([ta(end)/4 ta(end)/4],ylim0, 'color', 'k', 'linewidth', 1.5, 'linestyle', '-.')
        line([3*ta(end)/4 3*ta(end)/4],ylim0, 'color', 'k', 'linewidth', 1.5, 'linestyle', '-.')
        line([ta(end)/2 ta(end)/2],ylim0, 'color', 'k', 'linewidth', 1.5)
        box on
        
        subplot(size(Xobs_original,3), 2, (condition_cur-1)*2+2); hold on
        for kk = 1:size(X2,2)
            plot(ta, X2(:,kk), 'color', colort(condition_cur,:)*(kk-1)/size(X2,2), 'linewidth', 1.5 )
        end
        
        ylim1 = ylim;
        line([ta(end)/4 ta(end)/4],ylim1, 'color', 'k', 'linewidth', 1.5, 'linestyle', '-.')
        line([3*ta(end)/4 3*ta(end)/4],ylim1, 'color', 'k', 'linewidth', 1.5, 'linestyle', '-.')
        line([ta(end)/2 ta(end)/2],ylim1, 'color', 'k', 'linewidth', 1.5)
        box on

    end
end




%%



CP1 = model_cur{1,1}(:,hand_reorder_id);
CP1 = CP1 ./ sum(abs(CP1),2);

CP3 = condition_design*model_cur{1,3}(:,hand_reorder_id);

CP31 = abs(CP1) * abs(CP3)';

figure; hold on
scatter3(CP31(:,1),CP31(:,2),CP31(:,3))
line([0, max(max(CP31(:,1:2)))],[0, max(max(CP31(:,1:2)))], 'color', 'k', 'linewidth', 2)
%plot3(CP31(:,3),CP31(:,3),CP31(:,3))
xlabel('Vestibular')
ylabel('Both')
zlabel('Visual')
axis equal 
xlim([0, max(CP31(:))])
ylim([0, max(CP31(:))])
zlim([0, max(CP31(:))])
grid on


%%

oo = squeeze(offset_cur(:,1,:))*condition_design'/10;

figure; hold on
plot(exp(oo(:,1)), 'b')
plot(exp(oo(:,2)), 'm')
plot(exp(oo(:,3)), 'r')


%%

figure; hold on
scatter(1:size(oo,1),exp(oo(:,1))./sum(exp(oo),2), 'b','filled')
scatter(1:size(oo,1),exp(oo(:,2))./sum(exp(oo),2), 'm','filled')
scatter(1:size(oo,1),exp(oo(:,3))./sum(exp(oo),2), 'r','filled')

figure
scatter3(exp(oo(:,1))./sum(exp(oo),2),exp(oo(:,2))./sum(exp(oo),2),exp(oo(:,3))./sum(exp(oo),2))
xlabel('Vestibular')
ylabel('Both')
zlabel('Visual')
axis equal 

%%
figure
scatter3(exp(oo(:,1)),exp(oo(:,2)),exp(oo(:,3)))
xlabel('Vestibular')
ylabel('Both')
zlabel('Visual')
axis equal 
%%

oo = squeeze(offset_cur(:,1,:))*condition_design'/10;

figure; 
subplot(3,1,1); hold on;
plot(exp(oo(:,1))./sum(exp(oo),2), 'b')
plot(1/3+0*exp(oo(:,1))./sum(exp(oo),2), 'k')

subplot(3,1,2); hold on;
plot(exp(oo(:,2))./sum(exp(oo),2), 'm')
plot(1/3+0*exp(oo(:,1))./sum(exp(oo),2), 'k')

subplot(3,1,3); hold on;
plot(exp(oo(:,3))./sum(exp(oo),2), 'r')
plot(1/3+0*exp(oo(:,1))./sum(exp(oo),2), 'k')
%%

function circle_raster(spike_counts, rotation_direction, ccolor,rmax)
hold on


% Outter circle
r1 = 1;
x1 = linspace(-r1 ,r1 ,1000);
y1 = sqrt(r1.^2 -x1.^2);

% Inner Circle
r2 = 0.65;
x2 = linspace(-r2 ,r2 ,1000);
y2 = sqrt(r2.^2 -x2.^2);




axis equal
xlim([-1.2 1.2])
ylim([-0.2 1.2])

PSTH = mean(spike_counts, 2);
PSTH1 = PSTH(1:35)/rmax;
PSTH2 = PSTH(36:70)/rmax;

if strcmp(rotation_direction,'CCW')
    theta1 = linspace(0,pi, 35);
    theta2 = linspace(pi,0, 35);
    wratio1 = 1;
    wratio2 = r2/r1;
else
    theta1 = linspace(pi,0, 35);
    theta2 = linspace(0,pi, 35);
    rtmp = r1;
    r1 = r2;
    r2 = rtmp;
    wratio1 = r1/r2;
    wratio2 = 1;
end

% Arrows center
if strcmp(rotation_direction,'CW')
    centert1 = [r1, 0];
    centert2 = [-r2, 0];
else
    centert1 = [-r1, 0];
    centert2 = [r2, 0];
end

% Arrow start
xt1 = centert1(1) + [-0.05, 0 ,0.05];
yt1 = centert1(2) + [-0, -0.1 ,0];

% Arrow end
xt2 = centert2(1) + [-0.05, 0 ,0.05];
yt2 = centert2(2) + [-0, -0.1 ,0];


%XT = [r1*cos(theta1), r2*cos(theta2)];
%YT = [r1*sin(theta1), r2*sin(theta2)];
%UT = 1*[PSTH1'.*cos(theta1), PSTH2'.*cos(theta2)]/rmax;
%VT = 1*[PSTH1'.*sin(theta1), PSTH2'.*sin(theta2)]/rmax;
%q=quiver(XT,YT,UT,VT,'color', ccolor,'AutoScale','off', 'linewidth',6);
%q.ShowArrowHead = 'off';

q1 = quiver(r1*cos(theta1),r1*sin(theta1),PSTH1'.*cos(theta1)/rmax,PSTH1'.*sin(theta1)/rmax,...
    'color', ccolor,'AutoScale','off', 'linewidth',6*wratio1);

q2 = quiver(r2*cos(theta2),r2*sin(theta2),PSTH2'.*cos(theta2)/rmax,PSTH2'.*sin(theta1)/rmax,...
    'color', ccolor,'AutoScale','off', 'linewidth',6*wratio2);

q1.ShowArrowHead = 'off';
q2.ShowArrowHead = 'off';

% Outter circle
plot(x1,y1, 'k', 'linewidth',2)

% Inner circle
plot(x2,y2, 'k', 'linewidth',2)

% Arrow
patch(xt2,yt2,'k')

if strcmp(rotation_direction,'CW')
    text(-r1,-0.1, 'Start','FontSize',14,'HorizontalAlignment', 'center')
    line([r1, r2], [0, 0], 'color', 'k', 'linewidth',2, 'linestyle', '-')
else
    text(r1,-0.1, 'Start','FontSize',14,'HorizontalAlignment', 'center')
    line([-r1, -r2], [0, 0], 'color', 'k', 'linewidth',2, 'linestyle', '-')
end


axis off

end


function X = KhatriRaoProd(U,varargin)
%KR Khatri-Rao product.
%   kr(A,B) returns the Khatri-Rao product of two matrices A and B, of
%   dimensions I-by-K and J-by-K respectively. The result is an I*J-by-K
%   matrix formed by the matching columnwise Kronecker products, i.e.
%   the k-th column of the Khatri-Rao product is defined as
%   kron(A(:,k),B(:,k)).
%
%   kr(A,B,C,...) and kr({A B C ...}) compute a string of Khatri-Rao
%   products A o B o C o ..., where o denotes the Khatri-Rao product.
%
%   Version: 21/10/10
%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
if ~iscell(U), U = [U varargin]; end
K = size(U{1},2);
if any(cellfun('size',U,2)-K)
    error('kr:ColumnMismatch', ...
        'Input matrices must have the same number of columns.');
end
J = size(U{end},1);
X = reshape(U{end},[J 1 K]);
for n = length(U)-1:-1:1
    I = size(U{n},1);
    A = reshape(U{n},[1 I K]);
    X = reshape(bsxfun(@times,A,X),[I*J 1 K]);
    J = I*J;
end
X = reshape(X,[size(X,1) K]);
end






%%


