% Minimize the sum of the differences between obeserved data and estimates
% from Hill-based model
function Y = func_ii(par,X, y_obs)

N = size(X,1);
diff = (y_obs - Ymodel_ii(par,X)').^2;
Y = sum(diff)/N; 
end 


