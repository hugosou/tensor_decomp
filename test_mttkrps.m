%% Test the different ways of performing MTTKRP operations
% For a tensor X and factors A =[|A1, ..., AD|]
% MTTKRP(X,A,n) = X(n) * KhatriRao(AD, ..., An+1, An-1, ..., A1)
addpath(genpath('./'))


%% Tensor With Random Sparse Structure
clc
R = 30;
sizeX =  [5000,70,3,70];

% Random Factors
factors = cellfun(@(X,Y) rand(X,Y), ...
    num2cell(sizeX),...
    num2cell(repmat(R, 1,length(sizeX))),...
    'UniformOutput', false);

% Random Observed Data
observed = rand(sizeX);
observed = observed>0.9999;

% Random Observed Tensor
tensor = rand(sizeX);
tensor = tensor.*observed;

% Store only the observed values of the tensors
observed_compact = tensor(find(observed));
observed_indices = idten(size(tensor),find(observed));

mttkrp_sparse_tot = cell(1, length(sizeX));
mttkrp_custom_tot = cell(1, length(sizeX));

tic
for dimn = 1:length(sizeX)
    mttkrp_sparse_tot{dimn} = ...
        mttkrp_sparse(observed_compact, observed_indices, factors,dimn);
end
time_sparse = toc;


tic
for dimn = 1:length(sizeX)
    mttkrp_custom_tot{dimn} =...
        mttkrp_custom(tensor, factors, dimn);
end
time_vanilla = toc;


error = cellfun(@(X,Y) sum(abs(X(:)-Y(:))), ...
        mttkrp_sparse_tot, mttkrp_custom_tot);

disp('Test MTTKRP Sparse')
disp(['Error Tot : ', num2str(sum(error),2)])
disp(['T custom : ', num2str(time_vanilla,4)])
disp(['T sparse : ', num2str(time_sparse,4)])
disp(' ')

%% Tensor With Block Sparse Structure

observed_data = blkdiag(ones(400,400),ones(100,300),ones(800,100),ones(100,150),ones(100,150));
figure; 
imagesc(observed_data)
title('Block Observed Data'); colormap('gray')
xlabel('Dim. 1') ;ylabel('Dim. 3'); 

% Build observation tensor
D1 = size(observed_data ,1);
D2 = 70;
D3 = size(observed_data ,2);

observed_data = repmat(observed_data, [1,1, D2]);
observed_data = permute(observed_data, [1,3,2]);

% Random Observation Tensor
tensor = rand(D1,D2,D3);
tensor = tensor.*observed_data;

R = 30;

% Random Factors
factors= cellfun(@(X,Y) rand(X,Y), ...
    num2cell(size(tensor)), ...
    num2cell(repmat(R,1,ndims(tensor))), 'UniformOutput', false);

% Get Block indices
observed_data_compact = squeeze(observed_data(:,1,:));

id_bocks_d1 = find_x_blocks(observed_data_compact);
id_bocks_d3 = find_x_blocks(observed_data_compact');
num_blocks = size(id_bocks_d1,1);
assert(num_blocks == size(id_bocks_d3,1))
id_bocks_d2 = repmat([1,D2],num_blocks,1);


% Store Start and end point of each block
Xbindex_ste = [...
    mat2cell(id_bocks_d1, ones(1,num_blocks)),...
    mat2cell(id_bocks_d2, ones(1,num_blocks)),...
    mat2cell(id_bocks_d3, ones(1,num_blocks))];


% Store ids and values of each block
[block_observed, block_indices] = block_sparse(tensor,Xbindex_ste);

mttkrp_blocks_tot = cell(1, 3);
mttkrp_custom_tot = cell(1, 3);


tic
for dimn = 1:3
    mttkrp_blocks_tot{dimn} = ...
        mttkrp_block(block_observed, block_indices, factors,dimn);
end
time_blocks = toc;


tic
for dimn = 1:3
    mttkrp_custom_tot{dimn} =...
        mttkrp_custom(tensor, factors, dimn);
end
time_vanilla = toc;


error = cellfun(@(X,Y) sum(abs(X(:)-Y(:))), ...
        mttkrp_blocks_tot, mttkrp_custom_tot);

disp('Test MTTKRP Block Sparse')
disp(['Error Tot : ', num2str(sum(error),2)])
disp(['T custom : ', num2str(time_vanilla,4)])
disp(['T sparse : ', num2str(time_blocks,4)])
disp(' ')




