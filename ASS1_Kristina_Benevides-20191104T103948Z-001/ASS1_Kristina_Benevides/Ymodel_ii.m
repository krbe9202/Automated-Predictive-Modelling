
%Hill-based model
function Y = Ymodel_ii(par, X)
%par = [b0 b1 b2 b11 b22 b12 a0 a1 a2 a11 a22 a12];

%Calculate theta values as Ci/C where C is total concentration.
theta = zeros(size(X));
for i=1:length(X)
    theta(i,:) = X(i,:)./sum(X(i,:));
end

%Calculate IC50 and gamma.
IC50 = par(1) + par(2)*theta(:,1) + par(3)*theta(:,2) + par(4)*theta(:,1).^2 + par(5)*theta(:,2).^2 + par(6)*theta(:,1).*theta(:,2);
gamma = par(7) + par(8)*theta(:,1) + par(9)*theta(:,2) + par(10)*theta(:,1).^2 + par(11)*theta(:,2).^2 + par(12)*theta(:,1).*theta(:,2);

Y=zeros(size(X,1),1);
C=sum(X,2);
%Calculate dose effect levels using model (3) in article.
for i=1:length(gamma)
    Y(i) = 1/(1 +((sum(X(i,:))/IC50(i))^gamma(i)));
end

% Remove NaN values
Y(isnan(Y)) = 1;

end