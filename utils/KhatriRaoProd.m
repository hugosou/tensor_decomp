function X = KhatriRaoProd(U,Uind)
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
%   Adapated from Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)

R = size(U{1},2);


if nargin <2
    J = size(U{end},1);
    X = reshape(U{end},[J 1 R]);
    for n = length(U)-1:-1:1
        I = size(U{n},1);
        A = reshape(U{n},[1 I R]);
        X = reshape(bsxfun(@times,A,X),[I*J 1 R]);
        J = I*J;
    end
else
    J = length(Uind{end});
    X = reshape(U{end}(Uind{end},:),[J 1 R]);
    for n = length(U)-1:-1:1
        I = length(Uind{n});
        A = reshape(U{n}(Uind{n},:),[1 I R]);
        X = reshape(bsxfun(@times,A,X),[I*J 1 R]);
        J = I*J;
    end
end


X = reshape(X,[size(X,1) R]);



    
    

end
