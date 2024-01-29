function y = idten(nsizes,x)
% From linear indices to tensor indices
y = cell(1,length(nsizes));
[y{:}] = ind2sub(nsizes,x(:)); y = cell2mat(y);
end
