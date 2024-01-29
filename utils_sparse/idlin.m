function x = idlin(nsizes, y)
% From tensor indices to linear indices
x = 1 + [1,cumprod(nsizes(1:(end-1)))]*(y-1)';
end