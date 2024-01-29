function Dsnext = get_Ds_update(Ds, rho_D,W,Zs)

Wdims = size(W);
Ws = repmat(W,[ones(1,length(Wdims)),length(Wdims)]);
assert(all(size(Ws)==size(Zs)));
WmZs = Ws-Zs;

Dsnext = Ds + rho_D*WmZs;

end