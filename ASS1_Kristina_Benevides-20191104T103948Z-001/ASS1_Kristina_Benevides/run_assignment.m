
clear all; 
close all; 

%% Task 2

% Concentrations from Table I
% AG490
c1 = [0 0.3 1 3 10 30 100 300];
% U0126
c2 = [0 0.1 0.3 1 3 10 30 100];
% I-3-M
c3 = [0 0.3 1 3 10 30 100 300];

% Variables from Table II
%par = [b0 b1 b2 b11 b22 b12 a0 a1 a2 a11 a22 a12];
par = [117.11 -15.71 -86.15 79.55 42.67 -33.75 1.7 -1.02 0.41 0.31 -0.74 1.21]; 

% Make a grid over c1 and c2
[xa, xb] = ndgrid(linspace(0,300,100),linspace(0,100,100));
c1_2 = [xa(:), xb(:)]; 

%Calculate the effect-level with the Hill-based model in the Ymodel_ii func
Y1 = Ymodel_ii(par, c1_2); 
Y1 = reshape(Y1, [100,100]); 

%Plot the response surface
surf(linspace(0,300,100),linspace(0,100,100),Y1); 
xlabel('Dose level AG490 [µM]');
ylabel('Dose level U0126 [µM]'); 
zlabel('ATP-level'); 
title('Response surface of the Hill-based model'); 


%Store the 512 experiments in matrix X
[ca, cb, cc] = ndgrid(c1, c2, c3);
X = [ca(:), cb(:), cc(:)]; 

%Calculate effect-level for every experiment in X and add random noise
Y = Ymodel_ii(par,X);
y_observed = Y + normrnd(0,0.02,[size(Y),1]); 

%% Task 3

% The polynomial model 

%Store the dose levels in matrix z
for i=1:length(X)
z(:,i) = [1; X(i,1); X(i,2); X(i,3); X(i,1).*X(i,2); X(i,1).*X(i,3); X(i,2).*X(i,3); X(i,1).^2; X(i,2).^2; X(i,3).^2]; 
end

%Calculate the beta-parameters analytically and store them in a vector
 betas = inv(z*z')*z*y_observed;

%Use the parameter values from the analytical solution as starting guesses
%but add a factor of 0.9

% Estimate the beta-parameters with fminunc, store them in parameters_i and
% predict new effect-levels with these parameters using the polynomial model (Ymodel_i) 
f_i = @(par)func_i(par, X, y_observed');
x0_i=0.90*betas;
parameters_i = fminunc(f_i, x0_i);
Y_i = Ymodel_i(parameters_i,X);


% Estimate the b and a- parameters with fminunc. Predict new effect-levels with these
% using the Hill-based model (Ymodel_ii) 
f_ii = @(par)func_ii(par, X, y_observed');
x0_ii = 0.9*par; 
[parameters_ii] = fminunc(f_ii, x0_ii);
Y_ii=Ymodel_ii(parameters_ii,X);

% Estimate the w and alpha- parameters with fminunc. Predict new effect-levels with these
% using the simplified Hill-based model (Ymodel_iii) 
f_iii = @(par)func_iii(par, X, y_observed');
x0_iii = [0 0 0 0 1]; 
[parameters_iii] = fminunc(f_iii, x0_iii);
Y_iii=Ymodel_iii(parameters_iii,X);

%% Task 4

% 4a
% Plot

figure
subplot(1,3,1)
corr=corrcoef(Y_i, y_observed);
corr_sqr=corr(2)^2;
scatter(y_observed, Y_i,'.');
title('Polynomial model')
xlabel('Observed ATP level');
ylabel('Estimated ATP level');
text(0.5, 0.2, ['R^2 = ' num2str(corr_sqr)])

subplot(1,3,2)
corr=corrcoef(Y_ii, y_observed);
corr_sqr=corr(2)^2;
scatter(y_observed, Y_ii,'.'); 
title('Hill-based model')
xlabel('Observed ATP level');
ylabel('Estimated ATP level');
text(0.5, 0.2, ['R^2 = ' num2str(corr_sqr)])

subplot(1,3,3)
corr=corrcoef(Y_iii, y_observed);
corr_sqr=corr(2)^2;
scatter(y_observed, Y_iii,'.');
title('Simplified Hill-based model')
xlabel('Observed ATP level');
ylabel('Estimated ATP level');
text(0.5, 0.2, ['R^2 = ' num2str(corr_sqr)])

% 4b
% New dataset
Y_observed(:,1) = Ymodel_ii(par,X)';
Y_observed = Y_observed + normrnd(0,0.02,[size(Y_observed),1]); 

% Plot using the new external data

figure
subplot(1,3,1)
corr=corrcoef(Y_i, Y_observed);
corr_sqr=corr(2)^2;
scatter(Y_observed, Y_i,'.');
title('Polynomial model w external data')
xlabel('Observed ATP level');
ylabel('Estimated ATP level');
text(0.5, 0.2, ['R^2 = ' num2str(corr_sqr)])

subplot(1,3,2)
corr=corrcoef(Y_ii, Y_observed);
corr_sqr=corr(2)^2;
scatter(Y_observed, Y_ii,'.'); 
title('Hill-based model w external data')
xlabel('Observed ATP level');
ylabel('Estimated ATP level');
text(0.5, 0.2, ['R^2 = ' num2str(corr_sqr)])

subplot(1,3,3)
corr=corrcoef(Y_iii, Y_observed);
corr_sqr=corr(2)^2;
scatter(Y_observed, Y_iii,'.');
title('Simp. Hill-based model w external data')
xlabel('Observed ATP level');
ylabel('Estimated ATP level');
text(0.5, 0.2, ['R^2 = ' num2str(corr_sqr)])
%% Task 5

% Task 5a

%Calculate the relative error
R_org = norm(par - parameters_ii)/norm(par); 

% Task 5b

% Generate 99 more datasets and make Y_observed 512x100
for i=2:100
Y_observed(:,i) = Ymodel_ii(par,X)'; 
end

Y_observed = Y_observed + normrnd(0,0.02,[size(Y_observed),1]); 

%Store parameter estimates as column in Theta_hat for each of the 100
%datasets.
Theta_hat = zeros(12,100); 
errors = zeros(1,100); 
for i=1:100
f_Hill = @(par_Hill)func_ii(par_Hill, X, Y_observed(:,i)');
Theta_hat(:,i) = fminunc(f_Hill, x0_ii);
end
% use the rows in Theta_hat and calculate relative error for each of the
% Hill models parameters. 
errors = zeros(12,100); 
diffs = zeros(12,100); 
for i=1:12
diffs(i,:) = Theta_hat(i,:) - parameters_ii(i);  
errors(i) = norm(Theta_hat(i) - parameters_ii)/norm(Theta_hat(i)); 
end 
for i=1:100
    errors(1,i) = norm(diffs(1,i))/norm(Theta_hat(1,i)); 
    errors(2,i) = norm(diffs(2,i))/norm(Theta_hat(2,i)); 
    errors(3,i) = norm(diffs(3,i))/norm(Theta_hat(3,i)); 
    errors(4,i) = norm(diffs(4,i))/norm(Theta_hat(4,i)); 
    errors(5,i) = norm(diffs(5,i))/norm(Theta_hat(5,i)); 
    errors(6,i) = norm(diffs(6,i))/norm(Theta_hat(6,i)); 
    errors(7,i) = norm(diffs(7,i))/norm(Theta_hat(7,i)); 
    errors(8,i) = norm(diffs(8,i))/norm(Theta_hat(8,i)); 
    errors(9,i) = norm(diffs(9,i))/norm(Theta_hat(9,i)); 
    errors(10,i) = norm(diffs(10,i))/norm(Theta_hat(10,i)); 
    errors(11,i) = norm(diffs(11,i))/norm(Theta_hat(11,i));
    errors(12,i) = norm(diffs(12,i))/norm(Theta_hat(12,i));
end 
%Calculate iqr
IQR = zeros(12,1); 
for i=1:12
IQR(i)=iqr(errors(i,:)); 
end

%Plot iqr
figure
plot(IQR)
xlabel('Parameters'); 
ylabel('IQR'); 
title('IQR of the relative error');

%Boxplot of relative errors for IC50 and gamma parameters.
figure
suptitle('Relative errors for the b-parameters');
subplot(2,3,1)
boxplot(errors(1,:));
xlabel('b0')
subplot(2,3,2)
boxplot(errors(2,:));
xlabel('b1')
subplot(2,3,3)
boxplot(errors(3,:));
xlabel('b2')
subplot(2,3,4)
boxplot(errors(4,:));
xlabel('b11')
subplot(2,3,5)
boxplot(errors(5,:));
xlabel('b22')
subplot(2,3,6)
boxplot(errors(6,:));
xlabel('b12')

figure
suptitle('Relative error for the a-parameters');
subplot(2,3,1)
boxplot(errors(7,:));
xlabel('a0')
subplot(2,3,2)
boxplot(errors(8,:));
xlabel('a1')
subplot(2,3,3)
boxplot(errors(9,:));
xlabel('a2')
subplot(2,3,4)
boxplot(errors(10,:));
xlabel('a11')
subplot(2,3,5)
boxplot(errors(11,:));
xlabel('a22')
hold on;
subplot(2,3,6)
boxplot(errors(12,:));
xlabel('a12');



%% Task 6

%Start counter and set number of increases of batch size. Save the relative
%errors in row vector. 
count=1;
n = 3; %Use n = 3 or similar if you want to run the script quickly. 
rel_err=zeros(1,n);

%Initiate the matrices
x=[];
y_obs=[];

%For each loop the sequence of experiments gets twice as big and the IC50
%and gamma parameters are estimated for each experiment. The relative error 
%is calculated for each parameter estimate.   
while count<=n
   for i=1:count
       x=[x; X];
       y_obs = [y_obs; y_observed + normrnd(0,0.15,[size(y_observed),1])]; 
   end 
    
  fun = @(par)func_ii(par,x,y_obs'); 
   params_1 = fminunc(fun, x0_ii);
   rel_err(count)= norm(par - params_1)/norm(par);
   
   
   count = count + 1;
end 

% Plot the relative errors against number of experiments. 
axis_1=[];
for i=1:n
    axis_1=[axis_1 512*i];
end

figure
bar(axis_1, rel_err);
xlabel('# Experiments');
ylabel('Relative error');
title('Relative error for 512 drug combinations');

%% Task 7

% Repeat Task 6 but with fewer experiments


counts = 1; 
ns = 20; 
c1_fewer = [0 1 300]; 
c2_fewer = [0 3 100]; 
c3_fewer = [0 1 300]; 

[ca, cb, cc] = ndgrid(c1_fewer, c2_fewer, c3_fewer);
X_fewer = [ca(:), cb(:), cc(:)]; 
Y_fewer = Ymodel_ii(par,X_fewer); 

x_fewer = []; 
y_obs_fewer=[];

while counts <= n
   for i = 1:counts
       x_fewer=[x_fewer; X_fewer];
       y_obs_fewer=[y_obs_fewer; Y_fewer+normrnd(0, 0.08, 27, 1)];
   end 
   
   fun_fewer = @(par)func_ii(par, x_fewer, y_obs_fewer');
   params_fewer = fminunc(fun_fewer, x0_ii);
   rel_err_few(counts)= norm(par - params_fewer)/norm(par); 
   
   counts = counts +1; 
end 

figure
axis_fewer=[];
for i=1:n
    axis_fewer=[axis_fewer 27*i];
end
bar(axis_fewer, rel_err_few);
xlabel('# Experiments');
ylabel('Relative error');
title('Relative error for 27 drug combinations');