%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bayesian VAR version of Blanchard and Perotti
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
clc;

%% Bayesian VAR Version of Blanchard and Perotti 2002 paper

% Setup model parameters

priors.Lag = 4; % number of lags for FAVAR
priors.horizon = 21; % horizon for IRF
priors.keep = 10000; % Iterations to be considered
priors.burn = 20000; % Iterations to be discarded
priors.display = 1; % display iterations in command window
priors.K = 1; % No of govt spend components (total Spending)
priors.M = 2; % No of other macroeconomic variables in VAR (T,Y)
priors.N = priors.K + priors.M;
priors.constant = 1; % zero if no constant in the VAR
priors.trend = 2; % Trend component (Following deterministic trend as in BP 2002)
priors.quarters = 3;

% Import Data

data.raw1 = readtable('bp_data_ds.xlsx','Sheet',1);
data.raw = table2array(data.raw1(57:208,2:end));

% Transforming data
data.GDP = data.raw(:,4);
data.G = data.raw(:,3);
data.T = data.raw(:,8);
data.gtoy = mean(data.G./data.GDP);
data.ttoy = mean(data.T./data.GDP);
data.gtot = mean(data.G./data.T);
data.ttog = mean(data.T./data.G);


data.PGDP = data.raw(:,6);
data.POP = data.raw(:,7);
data.EGY = data.raw(priors.Lag+1:end,1);
data.ETY = data.raw(priors.Lag+1:end,2);
data.data1 = log([data.T./(data.PGDP.*data.POP) data.G./(data.PGDP.*data.POP) data.GDP./(data.PGDP.*data.POP)]);

% Effective sample size
data.T = size(data.data1,1) - priors.Lag; 

% the order of variable using only the output is: T, G, Y

%% Set up data (Deterministic Trend)

data.dataY = data.data1(priors.Lag+1:end,:);

data.quarters = repmat([1;2;3;4], data.T/4, 1)';

data.q1 = (data.quarters == 1);  % Q1 dummy
data.q2 = (data.quarters == 2);  % Q2 dummy
data.q3 = (data.quarters == 3);  % Q3 dummy
data.q4 = (data.quarters == 4);  % Q4 dummy

data.C = ones(1,data.T);

% Dummy to capture the large temporary tax rebate in 1975

data.dummy = zeros(1,data.T);
for i=0:4
    data.dummy(1, 58-i) = 1;
end

if priors.trend == 1  % linear trend
    data.dataX = [data.C', data.q1', data.q2', data.q3', data.dummy', (1:data.T)'];
elseif priors.trend == 2 % quadratic trend
    data.dataX = [data.C', data.q1', data.q2', data.q3', data.dummy', (1:data.T)', (1:data.T)'.^2]; 
end
for j=1:priors.Lag
    data.dataX = [data.dataX, data.data1(priors.Lag+1-j:end-j,:)];
end
data.Y = data.dataY;
data.X = data.dataX;

% Number of Variables is given by priors.N = 3

%% Bayesian Estimation (Minnesota Prior)

% No of coefficients (constant + 4 lags of four variables in each equation)

priors.ndt = 1 + priors.constant + priors.quarters + priors.trend; % no of columns before lagged variables 
priors.no_coef = priors.N * (priors.ndt + priors.N * priors.Lag);

% Setting up starting values and priors - see notes (tvar.pdf)

priors.phi0 = zeros(priors.N,priors.no_coef/priors.N); % Following Bernanke et al (2005)
priors.phi0 = priors.phi0(:); % vector form

priors.index = zeros(priors.N,priors.Lag);
for i=1:priors.N
    priors.index(i,:) = priors.ndt+i:priors.N:(priors.no_coef/priors.N);
end
priors.sigma = zeros(1*priors.N,1);

for q = 1:priors.N
    data.y1 = data.Y(:,q); % dependent variable
    data.x1 = data.X(:,[1:priors.ndt, priors.index(q,:)]); % constant + lags of dependent variable only
    priors.a = data.x1 \ data.y1; % regression coefficient
    priors.sigma(q,1) = ((data.y1 - data.x1*priors.a)'*(data.y1 - data.x1*priors.a))/(size(data.x1,1)-size(data.x1,2)); % error variance
end

priors.Q0 = diag(priors.sigma); % VAR var cov matrix

% Note: With univariate AR regressions, K < T and therefore we can use the
% above formula for variance computation (e'e/T-K)

priors.t0 = priors.N+2;

% Compute the prior hyperparameter H (var cov matrix of coefficients)

priors.lambda1 = 10;
priors.lambda2 = 1;
priors.lambda3 = 0;
priors.lambda4 = 50;

priors.H = zeros(priors.no_coef/priors.N,priors.N);

% Create an array of dimensions (N+N^2*lag) x N, which will contain the (N+N^2*lag) diagonal
% elements of the covariance matrix, in each of the N equations.

for i = 1:priors.N  % for each i-th equation
    for j = 1:(priors.no_coef/priors.N)   % for each j-th RHS variable in ith equation
        if j <= priors.ndt % j==1 if no trend 
            priors.H(j,i) = (priors.lambda4^2)*priors.sigma(i,1); % variance on constant, dummy for tax, quarterly dummies, and trends 
        elseif find(j==priors.index(i,:))>0
            ll = find(j==priors.index(i,:),2);
            priors.H(j,i) = (priors.lambda1/(ll^priors.lambda3))^2; % variance on own lags
        else
            for kj = 1:priors.N
                if find(j==priors.index(kj,:))>0
                    mm = kj;
                    nn = find(j==priors.index(kj,:),2);
                end
            end
            priors.H(j,i) = ((priors.lambda1*priors.lambda2)^2*priors.sigma(i,1))/(nn^(2*priors.lambda3)*priors.sigma(mm,1)); % variance on lags of other variables
        end
    end
end

% We get the prior variance covariance matrix H
priors.H0 = diag(priors.H(:));

% Initial values

draw.phivar = zeros(priors.N,priors.ndt+priors.N*priors.Lag);
priors.bols = (data.X\data.Y)';
priors.u = data.Y - data.X*priors.bols';

draw.phivar = priors.bols;
draw.Qvar = (priors.u'*priors.u)/(size(data.X,1)-size(data.X,2));

% Storage of results

results.phi = zeros(priors.N,(priors.no_coef/priors.N),priors.keep); % each phi drawn will be N x (1+N*lag)
results.Q = zeros(priors.N,priors.N,priors.keep); % each SIGMA drawn will be N x N

% IRF

results.IRF_G = zeros(priors.horizon,priors.N,priors.keep); % Govt Spending Shock
results.IRF_T = zeros(priors.horizon,priors.N,priors.keep); % Tax Shock

% GDP Multiplier

results.gdpmult_G = zeros(priors.horizon,1,priors.keep); 
results.gdpmult_T = zeros(priors.horizon,1,priors.keep);

%% Gibbs Sampling

c = 0;
for i = 1:priors.burn+priors.keep
    
    [draw.phivar] = BVAR_bp_phi(draw,priors,data);
    [draw.Qvar] = BVAR_bp_Sigma(draw,priors,data);
    [draw.IRF_G] = BVAR_bp_IRF_G(draw,priors,data);
    [draw.IRF_T] = BVAR_bp_IRF_T(draw,priors,data);
    
    if i > priors.burn
        c = c+1;
        results.phi(:,:,c) = draw.phivar;
        results.Q(:,:,c) = draw.Qvar;
        results.IRF_G(:,:,c) = draw.IRF_G;
        results.IRF_T(:,:,c) = draw.IRF_T;
    end
    
    if (mod(i,100)==0) && (priors.display == 1)
        disp(sprintf('Replication %s of %s' , num2str(i), num2str(priors.burn+priors.keep)));       
    end
end

%% Computing the posterior median and confidence intervals

posterior.phi = prctile(results.phi,[50 16 84],3);
posterior.Q = prctile(results.Q,[50 16 84],3);

posterior.IRF_G = prctile(results.IRF_G,[50 16 84],3);
posterior.IRF_T = prctile(results.IRF_T,[50 16 84],3);

%posterior.gdpmult_G = prctile(results.gdpmult_G,[50 16 84],3);
%posterior.gdpmult_T = prctile(results.gdpmult_T,[50 16 84],3);

%% Plot IRF with confidence interval and scale similar to the paper
% Government Spending Shock IRF

figure(1)
plot(posterior.IRF_G(1:20,1,1), 'LineWidth', 1.5, 'Color', 'b');  % Blue color for IRF
hold on;
plot(posterior.IRF_G(1:20,1,2), 'LineWidth', 1, 'LineStyle', '--', 'Color', 'g');  % Green color for lower bound
plot(posterior.IRF_G(1:20,1,3), 'LineWidth', 1, 'LineStyle', '--', 'Color', 'g');  % Green color for upper bound
line(xlim, [0 0], 'Color', 'k');  
hold off;

title('Response of Tax to Spending shock, DT');

% Define your desired x-ticks and y-ticks
xticks([1, 4, 7, 10, 13, 16, 19]);
yticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);

xlim([0, 21]);
ylim([-1.0, 2.5]);

figure(2)
plot(posterior.IRF_G(1:20,2,1), 'LineWidth', 1.5, 'Color', 'b');  % Blue color for IRF
hold on;
plot(posterior.IRF_G(1:20,2,2), 'LineWidth', 1, 'LineStyle', '--', 'Color', 'g');  % Green color for lower bound
plot(posterior.IRF_G(1:20,2,3), 'LineWidth', 1, 'LineStyle', '--', 'Color', 'g');  % Green color for upper bound
line(xlim, [0 0], 'Color', 'k');  
hold off;

title('Response of Spending to Spending shock, DT');

% Define your desired x-ticks and y-ticks
xticks([1, 4, 7, 10, 13, 16, 19]);
yticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);

xlim([0, 21]);
ylim([-1.0, 2.5]);

figure(3)
plot(posterior.IRF_G(1:20,3,1), 'LineWidth', 1.5, 'Color', 'b');  % Blue color for IRF
hold on;
plot(posterior.IRF_G(1:20,3,2), 'LineWidth', 1, 'LineStyle', '--', 'Color', 'g');  % Green color for lower bound
plot(posterior.IRF_G(1:20,3,3), 'LineWidth', 1, 'LineStyle', '--', 'Color', 'g');  % Green color for upper bound
line(xlim, [0 0], 'Color', 'k');  
hold off;

title('Response of GDP to Spending shock, DT');

% Define your desired x-ticks and y-ticks
xticks([1, 4, 7, 10, 13, 16, 19]);
yticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);

xlim([0, 21]);
ylim([-1.0, 2.5]);

%% Tax Shock IRF

figure(4)
plot(posterior.IRF_T(1:20,1,1), 'LineWidth', 1.5, 'Color', 'b');  % Blue color for IRF
hold on;
plot(posterior.IRF_T(1:20,1,2), 'LineWidth', 1, 'LineStyle', '--', 'Color', 'g');  % Green color for lower bound
plot(posterior.IRF_T(1:20,1,3), 'LineWidth', 1, 'LineStyle', '--', 'Color', 'g');  % Green color for upper bound
line(xlim, [0 0], 'Color', 'k');  
hold off;

title('Response of Tax to Tax shock, DT');

% Define your desired x-ticks and y-ticks
xticks([1, 4, 7, 10, 13, 16, 19]);
yticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]);

xlim([0, 21]);
ylim([-2.0, 1.0]);

figure(5)
plot(posterior.IRF_T(1:20,2,1), 'LineWidth', 1.5, 'Color', 'b');  % Blue color for IRF
hold on;
plot(posterior.IRF_T(1:20,2,2), 'LineWidth', 1, 'LineStyle', '--', 'Color', 'g');  % Green color for lower bound
plot(posterior.IRF_T(1:20,2,3), 'LineWidth', 1, 'LineStyle', '--', 'Color', 'g');  % Green color for upper bound
line(xlim, [0 0], 'Color', 'k');  
hold off;

title('Response of Spending to Tax shock, DT');

% Define your desired x-ticks and y-ticks
xticks([1, 4, 7, 10, 13, 16, 19]);
yticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]);

xlim([0, 21]);
ylim([-2.0, 1.0]);

figure(6)
plot(posterior.IRF_T(1:20,3,1), 'LineWidth', 1.5, 'Color', 'b');  % Blue color for IRF
hold on;
plot(posterior.IRF_T(1:20,3,2), 'LineWidth', 1, 'LineStyle', '--', 'Color', 'g');  % Green color for lower bound
plot(posterior.IRF_T(1:20,3,3), 'LineWidth', 1, 'LineStyle', '--', 'Color', 'g');  % Green color for upper bound
line(xlim, [0 0], 'Color', 'k');  
hold off;

title('Response of GDP to Tax shock, DT');

% Define your desired x-ticks and y-ticks
xticks([1, 4, 7, 10, 13, 16, 19]);
yticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]);

xlim([0, 21]);
ylim([-2.0, 1.0]);
