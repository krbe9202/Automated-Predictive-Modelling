
% Minimize the sum of the differences between obeserved data and estimates from polynomial
% model
function Y = func_i(par,X, y_obs)

N = size(X,1);
diff = (1/N).*(y_obs - Ymodel_i(par,X)').^2;
Y = sum(diff); 
end 