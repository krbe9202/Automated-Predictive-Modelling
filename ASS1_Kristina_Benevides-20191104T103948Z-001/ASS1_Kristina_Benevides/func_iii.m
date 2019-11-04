% Minimize the sum of the differences between obeserved data and estimates
% from simp. Hill model
function Y = func_iii(par,X, y_obs)

N = size(X,1);
diff = (y_obs - Ymodel_iii(par,X)').^2;
Y = sum(diff)/N; 
end 