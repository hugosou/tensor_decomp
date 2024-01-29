%% Load Fits to be Combined
data_folder = '/home/sou/Documents/Data/dresults/collapsed_CEK/';
fit_tot = cell(1);
fit_tot{1} = load([data_folder,'singlefit_standard_stitching0_flipback0_collapsed1_RSPdRSPgSC_2021_03_11_00_17.mat']);
fit_tot{2} = load([data_folder,'singlefit_standard_stitching0_flipback0_collapsed1_RSPdRSPgSC_2021_03_11_01_22.mat']);

%% Non Collapsed
fit_tot = cell(1);
fit_tot{1} = load('/home/sou/Documents/Data/dresults/missing_data_non_collapsed/singlefit_standard_stitching0_flipback0_collapsed0_RSPdRSPgSCSUBV1_2021_03_18_09_21.mat');
fit_tot{2} = load('/home/sou/Documents/Data/dresults/missing_data_non_collapsed/singlefit_standard_stitching0_flipback0_collapsed0_RSPdRSPgSCSUBV1_2021_03_18_12_09.mat');
fit_tot{3} = load('/home/sou/Documents/Data/dresults/missing_data_non_collapsed/singlefit_standard_stitching0_flipback0_collapsed0_RSPdRSPgSCSUBV1_2021_03_18_11_12.mat');
fit_tot{4} = load('/home/sou/Documents/Data/dresults/missing_data_non_collapsed/singlefit_standard_stitching0_flipback0_collapsed0_RSPdRSPgSCSUBV1_2021_03_18_11_41.mat');
fit_tot{5} = load('/home/sou/Documents/Data/dresults/missing_data_non_collapsed/singlefit_standard_stitching0_flipback0_collapsed0_RSPdRSPgSCSUBV1_2021_03_18_14_14.mat');
fit_tot{6} = load('/home/sou/Documents/Data/dresults/missing_data_non_collapsed/singlefit_standard_stitching0_flipback0_collapsed0_RSPdRSPgSCSUBV1_2021_03_18_16_18.mat');
fit_tot{7} = load('/home/sou/Documents/Data/dresults/missing_data_non_collapsed/singlefit_standard_stitching0_flipback0_collapsed0_RSPdRSPgSCSUBV1_2021_03_18_21_39.mat');

%% Collapsed

fit_tot = cell(1);
fit_tot{1} =load('/home/sou/Documents/Data/dresults/missing_data_collapsed/singlefit_standard_stitching0_flipback0_collapsed1_RSPdRSPgSCSUBV1_2021_03_21_03_19.mat');
fit_tot{2} =load('/home/sou/Documents/Data/dresults/missing_data_collapsed/singlefit_standard_stitching0_flipback0_collapsed1_RSPdRSPgSCSUBV1_2021_03_21_21_26.mat');
fit_tot{3} =load('/home/sou/Documents/Data/dresults/missing_data_collapsed/singlefit_standard_stitching0_flipback0_collapsed1_RSPdRSPgSCSUBV1_2021_03_21_21_47.mat');
fit_tot{4} =load('/home/sou/Documents/Data/dresults/missing_data_collapsed/singlefit_standard_stitching0_flipback0_collapsed1_RSPdRSPgSCSUBV1_2021_03_21_23_02.mat');
fit_tot{5} =load('/home/sou/Documents/Data/dresults/missing_data_collapsed/singlefit_standard_stitching0_flipback0_collapsed1_RSPdRSPgSCSUBV1_2021_03_22_01_39.mat');
fit_tot{6} =load('/home/sou/Documents/Data/dresults/missing_data_collapsed/singlefit_standard_stitching0_flipback0_collapsed1_RSPdRSPgSCSUBV1_2021_03_22_01_41.mat');
fit_tot{7} =load('/home/sou/Documents/Data/dresults/missing_data_collapsed/singlefit_standard_stitching0_flipback0_collapsed1_RSPdRSPgSCSUBV1_2021_03_22_03_19.mat');


%% Check that the same experimental and fit parameters were used 
experimental_parameters = fit_tot{1}.experimental_parameters;
fit_param_tot = fit_tot{1}.fit_param;

fit_str = {'Error','Check'};
dif_tot_ep = zeros(1,length(fit_tot)); 
dif_tot_fp = zeros(1,length(fit_tot)); 
for fit_cur = 2:length(fit_tot)
    experimental_parameters_test = fit_tot{fit_cur}.experimental_parameters;
    fit_param_tot_test = fit_tot{fit_cur}.fit_param;
   
    dfep = comp_struct(experimental_parameters,experimental_parameters_test,0);
    dffp = comp_struct(fit_param_tot,fit_param_tot_test,0);
    
    disp(['Fit#',num2str(fit_cur), ' EP:', fit_str{isempty(dfep)+1}, '/ FP:', fit_str{isempty(dffp)+1} ])
    dif_tot_ep(fit_cur) = not(isempty(dfep));
    dif_tot_fp(fit_cur) = not(isempty(dffp));
end
assert(all(dif_tot_ep==0), 'Experimental Parameters Mismatch')
assert(all(dif_tot_fp==0), 'Fit Parameters Mismatch')




%% Concatenate models and hyperparameters
hyptot = cell(size(fit_tot));
modtot = cell(size(fit_tot));
for fiti = 1:numel(fit_tot)
   hypi =  fit_tot{fiti}.hyperparams;
   modi =  fit_tot{fiti}.models_tot;
   
   % Check the fit structure
   msizes = size(modi);
   hsizes = cellfun(@(Z) length(Z) ,hypi)';
   
   
   assert(all(hsizes==msizes(1:end-1)),'Invalid number of hyperparameters')
   
   hyptot{fiti} = hypi;
   modtot{fiti} = modi;
end


[models_tot,hyperparams,measured] = gather_model(modtot,hyptot);



%% I NEED TO ADD WHAT HAPPENS WHEN DTOT AND STUFF


function [models,hyperparams,measured] = gather_model(modtot,hyptot)
% concatenate models modtot itted with hyperparameters hyptot


%assert(all(size(modtot)==size(hyptot)), 'Hyperparamaters and models should have the same size')
assert(all(diff(cellfun(@(Z) length(Z), hyptot))==0),'Invalid number of hyperparameters')
Ntests_tot = cellfun(@(Z) size(Z, ndims(Z)),modtot);
assert(all(diff(Ntests_tot)==0),'Invalid Number of Test Folder')
Ntests = Ntests_tot(1);

hlength = length(hyptot{1});

% Gather hyperparameters in a shared structure
hyperparams = cell(size(hyptot{1}));
for hypid = 1:hlength
    hypi = [];
    for fitid = 1:length(hyptot)
        hypi = [hypi, hyptot{fitid}{hypid}];
    end
    hypif = unique(hypi);
    hyperparams{hypid,1} = hypif;
end

% hypidx gives the index of each hyperparameters in the final structure
hypidx = hyptot;
for hypid = 1:hlength
    for fitid = 1:length(hyptot)
        hypcur = hyptot{fitid}{hypid};
        hypref = hyperparams{hypid};
        [~, loc] = ismember(hypcur, hypref);
        hypidx{fitid}{hypid} = loc;
    end
end

% Fill model_tot and is_measured
models = cell([cellfun(@(Z) length(Z),hyperparams)',Ntests]);
measured  = zeros(cellfun(@(Z) length(Z),hyperparams)');
for fitid = 1:length(hyptot)
    
    is_measured_cur =  zeros(cellfun(@(Z) length(Z),hyperparams)');
    is_measured_cur(hypidx{fitid}{:}) = 1;
    
    measured = measured  + is_measured_cur;
    models(hypidx{fitid}{:},:) = modtot{fitid};    
end

% Warnings
if not(isempty(find(measured(:)>1)));warning( 'Same model at least twice. Ambigous concatenation');end
if not(isempty((find(measured(:)==0)))); warning('Some model are not fitted'); end

end

function [df, match, er1, er2] = comp_struct(s1,s2,prt,pse,tol,n1,n2,wbf)
% check two structures for differances - i.e. see if strucutre s1 == structure s2
% function [match, er1, er2, erc, erv] = comp_struct(s1,s2,prt,pse,tol,n1,n2,wbf) 
% inputs  8 - 7 optional
% s1      structure one                              class structure
% s2      structure two                              class structure - optional
% prt     print test results (0 / 1 / 2 / 3)         class integer - optional
% pse     pause flag (0 / 1 / 2)                     class integer - optional
% tol     tol default tolerance (real numbers)       class integer - optional
% n1      first structure name (variable name)       class char - optional
% n2      second structure name (variable name)      class char - optional
% wbf     waitbar flag (0 / 1) default is 1          class integer - optional
%
% outputs 4 - 4 optional
% df      mis-matched fields with contents           class cell - optional
% match   matching fields                            class cell - optional
% er1     non-matching feilds for structure one      class cell - optional
% er2     non-matching feilds for structure two      class cell - optional
%
% prt:
%	0 --> no print
%	1 --> summary
%	2 --> print erros
%	3 --> print errors and matches
% pse:
%	1 --> pause for major erros
%	2 --> pause for all errors
%
% example:	[match, er1, er2] = comp_struct(data1,data2,1,1,1e-6,'data1','data2')
% michael arant - may 27, 2013
%
% updated - aug 22, 2013
%
% hint:
% passing just one structure causes the program to copy the structure
% and compare the two.  This is an easy way to list the structure
if nargin < 1; help comp_struct; error('I / O error'); end
if nargin < 2; s2 = s1; prt = 3; end
if nargin < 3 || isempty(prt); prt = 1; end
if nargin < 4 || isempty(pse); pse = 0; elseif pse ~= 1 && prt == 0; pse = 0; end
if nargin < 5 || isempty(tol); tol = 1e-6; end
if nargin < 6 || isempty(s1); n1 = 's1'; end
if nargin < 7 || isempty(s2); n2 = 's2'; end
if nargin < 8 || isempty(wbf); wbf = 1; end
if pse > prt, pse = prt; end
% solve
[match, er1, er2] = comp_struct_loop(s1,s2,prt,pse,tol,n1,n2,wbf);
% populate the error values
eval([char(n1) ' = s1;']);
eval([char(n2) ' = s2;']);
% size outputs
ner1 = numel(er1); ner2 = numel(er2);
% check that same number of errors were listed in each cell
if ner1 ~= ner2
	error(char('Something went very wrong in capturing errors.', ...
		'If possible, please email the two structures to moarant@gmail.com'));
else
	n = ner1;
end
% populate the error list
df = cell(n,3);
% loop the error lists
for ii = 1:n
	% capture the error text list
	temp1 = er1{ii}; temp2 = er2{ii};
	
	% see if the second structure exists
	if isempty(regexp(temp2,'is missing', 'once'))
		% record text error
		df{ii,1} = temp2;
		% see if matching structure 1 is missing (struture 2 listed as unique)
		if isempty(regexp(temp2,'is unique', 'once'))
			% unique to structure 2 - record value
			junk = regexp(temp2,' ','once');
			df{ii,3} = eval(temp2(1:junk-1));
		else
			% exists in 1 and 2 - evaluate types
			junk = regexp(temp2,' ','once'); temp2(junk:end) = [];
			junk = strfind(temp2,'.'); trash = temp2(junk+1:end); temp2(junk:end) = [];
			% if trash is empty, the field is a sub structure - list sub fields
			if isempty(trash)
				df{ii,3} = eval(['fieldnames(' temp2 ')'])';
			else
				% if numel(temp2) is > 1, then this is an indexed field
				% list the number if indexes and the type
				if (numel(eval(temp2)) - 1)
					df{ii,3} = sprintf('%s(#%g).%s is class %s', ...
						temp2,numel(eval(temp2)),trash,class(eval([temp2 '.' trash])));
				else
					% list the contents of the field
					df{ii,3} = eval([temp2 '.' trash]);
				end
			end
		end
	end
	if isempty(regexp(temp1,'is missing', 'once'))
		% record text error
		df{ii,1} = temp1;
		% see if matching structure 1 is missing (struture 2 listed as unique)
		if isempty(regexp(temp1,'is unique', 'once'))
			% unique to structure 2 - record value
			junk = regexp(temp1,' ','once');
			df{ii,2} = eval([temp1(1:junk-1)]);
		else
			% exists in 1 and 2 - evaluate types
			junk = regexp(temp1,' ','once'); temp1(junk:end) = [];
			junk = strfind(temp1,'.'); trash = temp1(junk+1:end); temp1(junk:end) = [];
			% if trash is empty, the field is a sub structure - list sub fields
			if isempty(trash)
				df{ii,2} = eval(['fieldnames(' temp1 ')'])';
			else
				% if numel(temp2) is > 1, then this is an indexed field
				% list the number if indexes and the type
				if (numel(eval(temp1)) - 1)
					df{ii,2} = sprintf('%s(#%g).%s is class %s', ...
						temp1,numel(eval(temp1)),trash,class(eval([temp1 '.' trash])));
				else
					% list the contents of the field
					df{ii,2} = eval([temp1 '.' trash]);
				end
			end
		end
	end
end
% optional text output
if prt
	fprintf('\n Error table\n');
	for ii = 1:n
		fprintf('\n%s    \n',df{ii,1});
		fprintf('Structure 1:  ');
		if isempty(df{ii,2}); fprintf('\n'); else; disp(df{ii,2}); end
		fprintf('Structure 2:  ');
		if isempty(df{ii,3}); fprintf('\n'); else; disp(df{ii,3}); end	
	end
	fprintf('\n\n\n\n\n');
end
end
%% recursive loop
function [match, er1, er2] = comp_struct_loop(s1,s2,prt,pse,tol,n1,n2,wbf)
% init outputs
match = {}; er1 = {}; er2 = {}; 
% test to see if both are structures
if isstruct(s1) && isstruct(s2)
	% both structures - get the field names for each structure
	fn1 = fieldnames(s1);
	fn2 = fieldnames(s2);
	% missing fields? get the common fields
	temp1 = ismember(fn1,fn2);
	temp2 = ismember(fn2,fn1);
	% missing fields in set 1
	for ii = find(~temp2)'
		er1{end+1} = sprintf('%s is missing field %s',n1,fn2{ii});
		er2{end+1} = sprintf('%s.%s is unique',n2,fn2{ii});
% 		er2{end+1} = s2.(fn2{ii});
		if prt > 1; fprintf('%s\n',er1{end}); end; if pse; pause; end
	end
	% missing fields in set 2
	for ii = find(~temp1)'
		er2{end+1} = sprintf('%s is missing field %s',n2,fn1{ii});
		er1{end+1} = sprintf('%s.%s is unique',n1,fn1{ii});
% 		er1{end+1} = s1.(fn1{ii});
		if prt > 1; fprintf('%s\n',er2{end}); end; if pse; pause; end
	end
	% index sizes match?  i.e. do both structures have the same # of indexes?
	inda = numel(s1); indb = numel(s2); inder = inda-indb;
	if inder < 0
		% struct 1 is smaller
		for ii = inda+1:indb
			er1{end+1} = sprintf('%s(%g) is missing',n1,ii);
			er2{end+1} = sprintf('%s(%g) is unique',n2,ii);
			if prt > 1; fprintf('%s\n',er1{end}); end; if pse; pause; end
		end
	elseif inder > 0
		% index 2 is smaller
		for ii = indb+1:inda
			er2{end+1} = sprintf('%s(%g) is missing',n2,ii);
			er1{end+1} = sprintf('%s(%g) is unique',n1,ii);
			if prt > 1; fprintf('%s\n',er2{end}); end; if pse; pause; end
		end
	end
	% get common fields
	fn = fn1(temp1); fnn = numel(fn); 
	% loop through structure 1 and match to structure 2
	ind = min([inda indb]); cnt = 0; 
	if wbf; wb = waitbar(0,'Comparing ....'); end
	for ii = 1:ind
		% loop each index
		for jj = 1:fnn
			% loop common field names
			if wbf; cnt = cnt + 1; waitbar(cnt/(ind*fnn),wb); drawnow; end
			% add index and field name to the structure name
			n1p = sprintf('%s(%g).%s',n1,ii,fn{jj});
			n2p = sprintf('%s(%g).%s',n2,ii,fn{jj});
			% recurse - run the program again on the sub-set of the structure
			[m e1 e2] = comp_struct_loop(s1(ii).(fn{jj}),s2(ii).(fn{jj}),prt,pse, ...
				tol,n1p,n2p,wbf);
			% add the sub-set (field name) results to the total results
			match = [match m']; 
			if ~isempty(e1) || ~isempty(e2)
				er1 = [er1 e1']; er2 = [er2 e2'];
			end
		end
	end
	if wbf;	close(wb); end
else
	% both are non-structures - compare
	% get the varable class and test
	c1 = class(s1); c2 = class(s2);
	if strcmp(c1,c2);
		% both are the same class
		if isequal(s1,s2)
			% results are equal
			match{end+1} = sprintf('%s and %s match',n1,n2);
			if prt == 3; fprintf('%s\n',match{end}); end
		else
			% same class but not equal
			% calculate error if type is single or double
			% test for function type match if function handle
			switch c1
				case {'single', 'double'}, 
					if numel(s1) ~= numel(s2) || size(s1,1) ~= size(s2,1)
						er = 1;
                    else
                        
                        er = 1;
						%er = norm(s1-s2);
					end
				case {'function_handle'},
					s1f = functions(s1); s2f = functions(s2);
					if strcmp(s1f.function,s2f.function)
						% same function with different values - record deviation and exit
						er = 0;
                        
						%er1{end+1} = sprintf('%s and %s are both %s but have different values', ...
						%	n1,n2,char(s1));
						%er2{end+1} = er1{end};
						if prt > 1; fprintf('%s\n',er1{end}); end;
						if pse > 1; pause; end
					else
						er = 1;
					end
				otherwise, er = 1;
			end
			% test error - error will be 0 (no error) or 1 (error) for all
			% classes except double and single.  double and single are the 
			% actual error which is tested against the tolerance
			% this was done for cases where structures are run on different 
			% platforms and numerical precision errors are observed
			if er > tol
				% sets do not match
				er1{end+1} = sprintf('%s and %s do not match',n1,n2);
				er2{end+1} = sprintf('%s and %s do not match',n2,n1);
				if prt > 1; fprintf('%s\n',er1{end}); end;
				if pse > 1; pause; end
			else
				% sets are a tolerance match
				match{end+1} = sprintf('%s and %s are tolerance match',n1,n2);
				if prt > 2; fprintf('%s\n',match{end}); end
			end
		end
	else
		% fields are different classes
		er1{end+1} = sprintf('%s is class %s, %s is class %s',n1,c1,n2,c2);
		er2{end+1} = sprintf('%s is class %s, %s is class %s',n2,c2,n1,c1);
		if prt > 1; fprintf('%s\n',er1{end}); end
		if pse; pause; end
	end
end
% transpose outputs
match = match'; er1 = er1'; er2 = er2';

end



















