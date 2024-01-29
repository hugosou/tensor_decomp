function ret = assertall(varargin)
% assertall - flexible multivalue assertions
%
% ASSERTALL(TESTS) raises an exception if any entry in TESTS is FALSE.  The
%   exception text indicates (by index) which of TESTS failed the assertion.
%
% ASSERTALL(COND, TESTS) raises an exception if any entry in TESTS violates
%   a condition indicated by the string COND, which may be:
%    'true' for the default behaviour
%    'false' for the reverse (exception if any of TESTS is TRUE) or
%    'eps' to raise an exception if any(abs(TESTS) > eps). The exact
%      tolerance may be set as an option.  For 'eps', the default is to show
%      the value of the failed TESTS rather than the indices.  To override
%      this set 'values' to [] and 'showIndices' to true.
%
% ASSERTALL(..., MSG) uses the exception text MSG instead of the default,
%   followed by the indices or values of the violation.
%
% ASSERTALL(..., {MSG, V1, V2, ...}) allows MSG to contain printf formatting
%   codes to be replaced by V1, ...
%
% OPTIONS:
% 'tolerance'	[eps]	set 'eps' condition threshold
% 'values'	[]	values to show for fails 
% 'showFails'	[10]	show only the first so many fails (or see above)
% 'showIndices'	[true]	show indices of fails
% 'showFailPerLine'	[5] show this many fails per line
% 'showFailSummary', [['']|max|maxabs|min|minabs|mean|meanabs|rms] summarise failed values thus
% 'warn'        [false] warn instead of raising exception
%
% See also: ASSERT, ERROR, TRY, MEXCEPTION

% maneesh
% 20170714: created
% 20171117: add warn option; showFails = max


% OPTIONS:
tolerance = eps;    % set 'eps' condition threshold
values = [];        % values to show for fails 
showFails = Inf;     % show only the first so many fails
showIndices = true; % show indices of fails
showFailsPerLine = 5; % show this many fails per line
showFailSummary = ''; % summarise failed values
warn = false;       % warn instead of raising exception
% OPTIONS DONE

ret = 1;
cond = 'true';
msg = {};

%% sort out arguments 
if nargin == 1                          % assertall(data)
  data = varargin{1};   varargin = {};
else
  if ischar(varargin{1})                % assertall(cond, data, ...)
    if nargin == 2                      % assertall(cond, data);
      [cond, data] = deal(varargin{1:2});   varargin = {};
    else                                % assertall(cond, data, msg, ...)
      [cond, data, msg] = deal(varargin{1:3}); 
      varargin(1:3) = [];
    end
  else                                  % assertall(data, msg, ...)
    [data, msg] = deal(varargin{1:2});
    varargin(1:2) = [];
  end
end

%% set some conditional defaults
switch(cond)
 case 'eps', 
  showIndices = false;
  values = data;
end

%% options
optlistassign('ignorecase', who, varargin);

%% process message argument
if isempty(msg)
  msg = 'Assertion failed';
end
if ischar(msg)
  msg = {msg};
end


  
switch(cond)
 case 'true', 
  fails = find(~data);
 case 'false', 
  fails = find(data);
 case 'eps', 
  fails = find(abs(data) > tolerance);
 otherwise,
  error('unrecognised assertion condition');
end

  
if ~isempty(fails)
  errmsg = sprintf(msg{:});

  if (showFails == 0)

    moremsg = sprintf(' at %d place(s)', length(fails));

  elseif (showFails > 0)

    if length(fails) > showFails
      moremsg = sprintf(' and at %d more place(s)', length(fails)-showFails);
    else
      moremsg = '';
      showFails = length(fails);
    end
    
    % find index subscripts
    [indx{1:ndims(data)}] = ind2sub(size(data), fails);
    indxcat = cat(2,indx{:});
      
    ii = 1;
    while(ii <= min(showFails, length(fails)))
      if showFails > showFailsPerLine
        errmsg = [errmsg, '\n'];
      end
      failmsg = '\t';
      for jj = 1:showFailsPerLine
        if showIndices
          failmsg = [failmsg, mat2str(indxcat(ii, :)), ' '];
        end

        if ~isempty(values)
          failmsg = [failmsg, num2str(values(fails(ii))), '\t'];
        end

        ii = ii + 1;
        if (ii > min(showFails, length(fails)))
          break;
        end
      end
      errmsg = sprintf('%s%s', errmsg, failmsg);
    end
  end
  errmsg = [errmsg, moremsg];
  
  if (~isempty(showFailSummary))
    
    if isa(showFailSummary, 'function_handle')
      moremsg = sprintf(' (summary violation = %g)', feval(showFailSummary, values(fails)));
    else
      switch (showFailSummary)
       case {'max', 'min', 'maxabs', 'minabs', 'sum', 'mean', 'rms'}
        moremsg = sprintf(' (%s violation = %g)', showFailSummary, ...
                          feval(showFailSummary, values(fails))); 
       case 'meanabs', 
        moremsg = sprintf(' (%s violation = %g)', showFailSummary, mean(abs(values(fails))));
      end
    end
  
    errmsg = [errmsg, moremsg];
  end
  
  if (warn)
    warning('AssertAll:failed', errmsg);
    ret = 0;
  else
    ME = MException('AssertAll:failed', errmsg);
    throwAsCaller(ME);
  end
end



