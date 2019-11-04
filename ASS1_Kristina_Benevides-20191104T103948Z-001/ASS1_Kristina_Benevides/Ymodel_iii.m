
%Simplified Hill-model
function Y = Ymodel_iii(par, X)
%par = [w0 w1 w2 w3 alpha]
N=size(X,1);
Y=zeros(size(X,1),1);

for i=1:N    
    Y(i)=1/((1 + par(1) +par(2)*X(i,1) + par(3)*X(i,2) + par(4)*X(i,3) )^par(5));
end

end