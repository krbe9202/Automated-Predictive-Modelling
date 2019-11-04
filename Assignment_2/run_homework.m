
clear all; 
close all;

%% Task 1

%a 

%Generate uniformly distributed random numbers using formula 
%r = a + (b-a).*rand(N,1) where [a,b] = [-1,1] in this case and
%N_train = 90.

N_train = 90; 
noOfDims=500;
X_train = zeros(N_train,noOfDims); 

for i= 1:N_train
    x = -1 + 2*rand(noOfDims,1);
    X_train(i,:) = x; 
end 

%Create weight vectors for the three different models
w1 = zeros(noOfDims,1); 
w1(1:500) = 1/500; 

w2 = zeros(noOfDims,1); 
w2(1:10) = 1/10; 

w3 = zeros(noOfDims,1); 
for i=1:noOfDims
    w3(i) = 2.^(-i);  
end 

%Generate noisy observations

y1 = X_train*w1; 
y1 = y1 + normrnd(0,0.00005,[size(y1),1]);

y2 = X_train*w2; 
y2 = y2 + normrnd(0,0.04,[size(y2),1]); 

y3 = X_train*w3; 
y3 = y3 + normrnd(0,0.07,[size(y3),1]);

Yobs_train = [y1 y2 y3]; 

%b 

%Repeat Task 1a for N_external = 1000.

N_external = 1000; 
X_external = zeros(N_external,noOfDims); 

for i= 1:N_external
    x = -1 + 2*rand(noOfDims,1);
    X_external(i,:) = x; 
end 

y1_ext = X_external*w1; 
y1_ext = y1_ext + normrnd(0,0.0005,[size(y1_ext),1]);

y2_ext = X_external*w2; 
y2_ext = y2_ext + normrnd(0,0.04,[size(y2_ext),1]); 

y3_ext = X_external*w3; 
y3_ext = y3_ext + normrnd(0,0.07,[size(y3_ext),1]);

Yobs_external = [y1_ext y2_ext y3_ext]; 

%% Task 2

%a

%w_ML = olsregression(X_train,Yobs_train); 

%b 

lamda = 0:0.01:0.99; 
k = 9; 
N = size(X_train,1); 

wRR_1 = zeros(noOfDims, numel(lamda));
wRR_2 = zeros(noOfDims, numel(lamda));
wRR_3 = zeros(noOfDims, numel(lamda));

%Cross validation
for i=1:numel(lamda)
    for j=1:k

    c = cvpartition(size(X_train,1),'KFold',k); 

    ind_train = training(c,1); 
    ind_test = test(c,1); 

    X_train_RR = X_train(ind_train,:); 
    X_test_RR = X_train(ind_test,:); 

    Yobs_train_RR1 = Yobs_train(ind_train,1);
    Yobs_test_RR1 = Yobs_train(ind_test,1); 
    
    Yobs_train_RR2 = Yobs_train(ind_train,2);
    Yobs_test_RR2 = Yobs_train(ind_test,2); 

    Yobs_train_RR3 = Yobs_train(ind_train,3);
    Yobs_test_RR3 = Yobs_train(ind_test,3); 

    %Model 1
        wRR_1(:,i) = ridgereg(X_train_RR,Yobs_train_RR1,lamda(i)); 
        y_test_1_hat_RR = X_test_RR*wRR_1(:, i);
          
    %Model 2
        wRR_2(:,i) = ridgereg(X_train_RR,Yobs_train_RR2,lamda(i)); 
        y_test_2_hat_RR = X_test_RR*wRR_2(:, i);
    
    %Model 3
        wRR_3(:,i) = ridgereg(X_train_RR,Yobs_train_RR3,lamda(i)); 
        y_test_3_hat_RR = X_test_RR*wRR_3(:, i);
    

    end
    RMSE_1(i) = sqrt(mean((Yobs_test_RR1 - y_test_1_hat_RR).^2));
    RMSE_2(i) = sqrt(mean((Yobs_test_RR2 - y_test_2_hat_RR).^2));
    RMSE_3(i) = sqrt(mean((Yobs_test_RR3 - y_test_3_hat_RR).^2));
    
end 

%Lowest value for RMSE --> best lambda value
[minVal1,iMin1_RR_test]=min(RMSE_1);
[minVal2,iMin2_RR_test]=min(RMSE_2);
[minVal3,iMin3_RR_test]=min(RMSE_3);

lambdaBest1=lamda(iMin1_RR_test);
lambdaBest2=lamda(iMin2_RR_test);
lambdaBest3=lamda(iMin3_RR_test);

wRR_1_best = ridgereg(X_train,Yobs_train(:,1),lambdaBest1); 
wRR_2_best = ridgereg(X_train,Yobs_train(:,2),lambdaBest2);
wRR_3_best = ridgereg(X_train,Yobs_train(:,3),lambdaBest3);
 
figure
plot([w1 wRR_1_best])
title('Model #1');
figure
plot([w2(1:100) wRR_2_best(1:100)])
title('Model #2');
figure
plot([w3(1:100) wRR_3_best(1:100)])
title('Model #3');

figure
subplot(1,3,1)
plot(lamda, RMSE_1); 
title('Model #1'); 
xlabel('lambda')
grid on; 
subplot(1,3,2)
plot(lamda, RMSE_2); 
title('Model #2'); 
grid on;
subplot(1,3,3)
plot(lamda, RMSE_3); 
title('Model #3'); 
grid on;

y_external_1_hat_RR = X_external*wRR_1_best; 
y_external_2_hat_RR = X_external*wRR_2_best; 
y_external_3_hat_RR = X_external*wRR_3_best; 


RMSE_RR_Test1_external = sqrt(mean((Yobs_external(:,1) - y_external_1_hat_RR).^2));
RMSE_RR_Test2_external = sqrt(mean((Yobs_external(:,2) - y_external_2_hat_RR).^2));
RMSE_RR_Test3_external = sqrt(mean((Yobs_external(:,3) - y_external_3_hat_RR).^2));

%Plot scatterplots
figure 
subplot(1,3,1)
corr=corrcoef(y_external_1_hat_RR, Yobs_external(:,1));
corr_sqr=corr(2)^2;
plot(y_external_1_hat_RR, Yobs_external(:,1),'.');
title('Model #1')
xlabel('Estimated response');
ylabel('Observed response');
text(0, -0.06, ['R^2 = ' num2str(corr_sqr)])

subplot(1,3,2)
corr=corrcoef(y_external_2_hat_RR, Yobs_external(:,2));
corr_sqr=corr(2)^2;
plot(y_external_2_hat_RR, Yobs_external(:,2),'.'); 
title('Model #2')
xlabel('Estimated response');
ylabel('Observed response');
text(0.1, -0.45, ['R^2 = ' num2str(corr_sqr)])

subplot(1,3,3)
corr=corrcoef(y_external_3_hat_RR, Yobs_external(:,3));
corr_sqr=corr(2)^2;
plot(y_external_3_hat_RR, Yobs_external(:,3),'.'); 
title('Model #3')
xlabel('Estimated response');
ylabel('Observed response');
text(0.1, -0.8, ['R^2 = ' num2str(corr_sqr)]); 

%% Task 3
%  
% %a and b 
reps = 100; 

%Shuffle the X_train and Yobs_train matrix
r = randperm(size(X_train,1)); 
shuffled_X_train = X_train(r,:); 
shuffled_Yobs_train = Yobs_train(r,1);
%Initiate matrices
RMSEavg_inner = []; 
RMSE_outer = []; 
wRR_opt = []; 

%Repeat a) a 100 times. 
for h = 1:reps 
for j=1:k    
    for i=1:numel(lamda)
    c = cvpartition(size(X_train,1),'KFold',k); 

    ind_train = training(c,1); 
    ind_test = test(c,1); 

    X_train_RR = shuffled_X_train(ind_train,:); 
    X_test_RR = shuffled_X_train(ind_test,:); 

    Yobs_train_RR1 = shuffled_Yobs_train(ind_train,1);
    Yobs_test_RR1 = shuffled_Yobs_train(ind_test,1); 
    
        for n= 1:k-1
        c_inner = cvpartition(size(X_train_RR,1),'KFold',k+1); 
        ind_train_inner = training(c_inner,1); 
        ind_test_inner = test(c_inner,1); 

        X_train_RR_inner = X_train_RR(ind_train_inner,:); 
        X_test_RR_inner = X_train_RR(ind_test_inner,:); 

        Yobs_train_RR1_inner = Yobs_train_RR1(ind_train_inner,1);
        Yobs_test_RR1_inner = Yobs_train_RR1(ind_test_inner,1); 

        wRR_inner(:,i) = ridgereg(X_train_RR_inner,Yobs_train_RR1_inner,lamda(i)); 
        y_test_1_hat_inner = X_test_RR_inner*wRR_inner(:, i);
        RMSE_inner(i,n) = sqrt(mean((Yobs_test_RR1_inner - y_test_1_hat_inner).^2)); 
        end
        
        %Calculate the smallest average RMSE from inner loop and use that 
        %to calculate the best weight vectors. 
        RMSE_best_inner(j) = min(mean(RMSE_inner(i,:)));
        [RMSE_best_inner_val,iMin_RR_inner]=min(RMSE_best_inner);
        lambdaOpt=lamda(iMin_RR_inner);
        wRR_best_inner(:,j) = ridgereg(X_train_RR,Yobs_train_RR1,lambdaOpt); 
         
        
        wRR_1(:,i) = ridgereg(X_train_RR,Yobs_train_RR1,lamda(i)); 
        y_test_1_hat_RR = X_test_RR*wRR_1(:, i); 
        RMSEall_outer(i,j) = sqrt(mean((Yobs_test_RR1 - y_test_1_hat_RR).^2));
        
%       y_hat_outer = X_test_RR*wRR_best_inner(:,j); 
%       RMSEall_outer(j) = sqrt(mean((Yobs_test_RR1 - y_hat_outer).^2));
    end  
        RMSE_best_outer(j) = min(sum(RMSEall_outer(i,:)));

end
RMSEavg_inner = [RMSEavg_inner RMSE_best_inner];
RMSE_outer = [RMSE_outer  RMSE_best_outer]; 
wRR_opt = [wRR_opt wRR_best_inner]; 

end

%Plot histograms
figure
subplot(1,2,1)
histogram(RMSEavg_inner); 
title('Average RMSE for inner fold');
xlabel('RMSE'); 

subplot(1,2,2)
histogram(RMSE_outer);
title('RMSE for outer fold');
xlabel('RMSE'); 

figure
histogram(RMSEavg_inner); 
title('Average RMSE for inner fold');
xlabel('RMSE'); 
hold on; 
histogram(RMSE_outer);
title('RMSE for both folds');
xlabel('RMSE'); 


%c 

%Create external set. 
N_bigexternal = 4000; 
X_bigexternal = zeros(N_bigexternal,noOfDims); 

for i= 1:N_bigexternal
    x = -1 + 2*rand(noOfDims,1);
    X_bigexternal(i,:) = x; 
end 

y_bigexternal = X_bigexternal*w1; 
y_bigexternal= y_bigexternal + normrnd(0,0.0005,[size(y_bigexternal),1]);

y_bigexternal_1_hat_RR = X_bigexternal*wRR_opt; 

RMSE_big = sqrt(mean((y_bigexternal - y_bigexternal_1_hat_RR).^2));

%Plot histogram
figure
histogram(RMSE_big)
title('RMSE big'); 
xlabel('RMSE'); 
