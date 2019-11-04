
clear all; 
close all; 
%% Task 2
tic
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

% Perform k-means clustering 
k_num = 30; 
idx = kmeans(X,k_num); 

% Choose one example from every cluster and store in X_train
X_train = [];  
for i=1:k_num
       cluster = X(idx == i,:);
       r = randperm(size(cluster,1));
       cluster = cluster(r,:); 
       X_train_i = cluster(1,:);
       X_train = [X_train; X_train_i]; 
end  

% Create observations from Hill-based model
setY = Ymodel_ii(par,X_train);
Y_train = setY + normrnd(0,0.02,[size(setY),1]);


%Bootstrap
% b = 10;
% X_train_b = []; 
% for i = 1:b
% X_train_b = [X_train_b; datasample(X_train,k_num)]; 
% end

%Neural network

% 10-fold cross validation to decide the optimal number of hidden nodes
% in the set V = 1,2,3,5,10
k = 10; 

V = [1,2,3,5,10]; 
numNN = 10; 
NN = cell(1,numNN); 
perf_avg = [];
Nmax = 5; 
n = 1; 


while n <= Nmax
    c_qbc = cvpartition(size(X_train,1),'KFold',k); 

    ind_train = training(c_qbc,1); 
    ind_test = test(c_qbc,1); 

    XsetTrain = X_train(ind_train,:); 
    XsetTest = X_train(ind_test,:); 

    YsetTrain = Y_train(ind_train,:); 
    YsetTest = Y_train(ind_test,:); 
        for j = 1:length(V)
            for i=1:k
            net = feedforwardnet(V(j));
            net.layers{2}.transferFcn = 'logsig';
            net.divideParam.trainRatio = 0.8; %Set training set to 80 %
            net.divideParam.valRatio = 0.2;   %Set validation set to 20%
            net.divideParam.testRatio = 0;    %Set test set to 0 % to use the holdout XsetTest for testing
            NN{i,j} = train(net,XsetTrain',YsetTrain'); 
            y = NN{i,j}(XsetTest'); 
            perf(i) = perform(net,YsetTest',y);
            end
        perf_avg(j) = mean(perf); 
        end
    
     % Choose H- value in set V which corresponds to the smallest mse and
     % use those nets which corresponds to that H-value. 
     [valHidden, numHiddenInd] = min(perf_avg); 
     numHidden = V(numHiddenInd); 
     nets = NN(:,numHiddenInd); 
     
     % Predict response values using the nets with the optimal number of
     % hidden nodes. 
    for h=1:k
        for m=1:size(X,1) 
        netH = nets{h}; 
        y_response(m,h) = netH(X(m,:)'); 
        end
    end
    
    % Calculate the standard deviation for every prediction
    for p=1:size(X,1)
        stds(p) = std(y_response(p,:)); 
    end
    
    %Pick out the 30 largest standard deviations
    [biggest_std, std_ind] = maxk(stds,30);
    
    % Exit loop if largest std. is less than 0.05 or num. of iterations
    % exceeds maximum number of iterations
    if(biggest_std(1) <= 0.05 || n >= Nmax) 
    break; 
    end
    
    %Extend training set with the 30 examples corresponding to the highest standard deviation 
    X_train = [X_train; X(std_ind,:)]; 
    Y_train_new = Ymodel_ii(par,X_train);
    Y_train = Y_train_new + normrnd(0,0.02,[size(Y_train_new),1]); 
    n = n+1; 
end



%% Task 4

% Make a grid over c1 and c2 (keeping c3 to 0) with 10000 points. 

c = [c1_2 , zeros(10000,1)]; 

%Calculate the effect-level with the Hill-based model in the Ymodel_ii func
Y2 = Ymodel_ii(par, c); 
Y2 = Y2 + normrnd(0,0.02,[size(Y2),1]); 

% QBC

  for a=1:k
        for b=1:size(c,1) 
        netH_big = nets{a}; 
        ybig_response(b,a) = netH_big(c(b,:)'); 
        end
  end
  
%Caluclate RMSE for the large data set. 
  RMSE_qbc = sqrt(mean((Y2 - ybig_response).^2)); 
%   
% % Random

% Repeat the cross validation performed for QBC but use a random subset of
% X and Y

r = randperm(size(X,1)); 
shuffled_X = X(r,:);
shuffled_Y = Y(r,:); 
randX = shuffled_X(1:150,:); 
randY = shuffled_Y(1:150,:); 

c_random = cvpartition(size(randX,1),'KFold',k); 

ind_train_random = training(c_random,1); 
ind_test_random = test(c_random,1);

XsetTrainRandom = randX(ind_train_random,:); 
XsetTestRandom = randX(ind_test_random,:); 

YsetTrainRandom = randY(ind_train_random,:); 
YsetTestRandom = randY(ind_test_random,:); 

 for t = 1:length(V)
        for s=1:k
        net_random = feedforwardnet(V(t));
        net_random.layers{2}.transferFcn = 'logsig';
        net_random.divideParam.trainRatio = 0.8; 
        net_random.divideParam.valRatio = 0.2;
        net_random.divideParam.testRatio = 0;
        NN_random{s,t} = train(net_random,XsetTrainRandom',YsetTrainRandom'); 
        y = NN_random{s,t}(XsetTestRandom'); 
        perf_random(s) = perform(net_random,YsetTestRandom',y);
        end
        perf_avg_random(t) = mean(perf_random); 
 end
    
     [valHidden, numHiddenInd_random] = min(perf_avg_random); 
     numHidden_random = V(numHiddenInd_random); 
     nets_random = NN_random(:,numHiddenInd_random); 

   for a=1:k
        for b=1:size(c,1) 
        netH_big_random = nets_random{a}; 
        ybig_response_random(b,a) = netH_big_random(c(b,:)'); 
        end
   end
      RMSE_random = sqrt(mean((Y2 - ybig_response_random).^2)); 
      
% Optimal design 

% Repeat the cross validation performed for QBC but use candexch to select optimal subset of
% X and Y

R_opt = candexch(X, 150); % This however, only selects three experiments, which I believe leads to the optimal design having the highest RMSE. 
optX = X(R_opt,:); 
optY = Y(R_opt,:); 

c_opt = cvpartition(size(optX,1),'KFold',k); 

ind_train_opt = training(c_opt,1); 
ind_test_opt = test(c_opt,1);

XsetTrainOpt = optX(ind_train_opt,:); 
XsetTestOpt = optX(ind_test_opt,:); 

YsetTrainOpt = optY(ind_train_opt,:); 
YsetTestOpt = optY(ind_test_opt,:); 

 for t = 1:length(V)
        for s=1:k
        net_opt = feedforwardnet(V(t));
        net_opt.layers{2}.transferFcn = 'logsig';
        net_opt.divideParam.trainRatio = 0.8; 
        net_opt.divideParam.valRatio = 0.2;
        net_opt.divideParam.testRatio = 0;
        NN_opt{s,t} = train(net_opt,XsetTrainOpt',YsetTrainOpt'); 
        y = NN_opt{s,t}(XsetTestOpt'); 
        perf_opt(s) = perform(net_opt,YsetTestOpt',y);
        end
        perf_avg_opt(t) = mean(perf_opt); 
    end
    
     [valHidden, numHiddenInd_opt] = min(perf_avg_opt); 
     numHidden_random = V(numHiddenInd_opt); 
     nets_opt = NN_opt(:,numHiddenInd_opt); 

   for a=1:k
        for b=1:size(c,1) 
        netH_big_opt = nets_opt{a}; 
        ybig_response_opt(b,a) = netH_big_opt(c(b,:)'); 
        end
  end
      RMSE_opt = sqrt(mean((Y2 - ybig_response_opt).^2)); 


%Plot RMSE
meanRmseQBC = mean(RMSE_qbc); 
meanRmseRandom = mean(RMSE_random); 
meanRmseOptimal = mean(RMSE_opt);
RMSE_values = [meanRmseQBC meanRmseRandom meanRmseOptimal];
names = categorical({'QBC','RANDOM','OPTIMAL'});

figure
bar(names, RMSE_values);
ylabel('RMSE'); 
title('mean RMSE values for the three approaches')
toc