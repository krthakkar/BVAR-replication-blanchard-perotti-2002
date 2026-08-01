function [IRF_G] = BVAR_bp_IRF_G(draw,priors,data)

Y = data.Y; % T x N
X = data.X; % T x (ndt terms + N*lag)

N = priors.N;
T = data.T;
P = priors.Lag;
ndt = priors.ndt;

ETY = data.ETY;

gtoy = data.gtoy;  
gtot = data.gtot;  

% Extract deterministic and lag coefficients
PHI0 = draw.phivar(:,1:ndt);

PHI = cell(P,1);
for k = 1:P
    PHI{k} = draw.phivar(:, ndt+1+(k-1)*N : ndt+k*N); 
end

% Reduced-form residuals
u = Y - X * draw.phivar';

% u is T x N

% B&P (2002) identification: spending ordered first

a1 = 2.08; % Average net-tax elasticity used for IRFs
b1 = 0;    % Automatic spending elasticity
b2 = 0;    % Spending ordered before taxes

% Cyclically adjusted reduced-form residuals
cyclical = zeros(T,2);

cyclical(:,1) = u(:,1) - ETY .* u(:,3); % adjusted tax residual
cyclical(:,2) = u(:,2);                 % adjusted spending residual

% Estimate a2:
% adjusted taxes on constant and adjusted spending
a = [ones(T,1), cyclical(:,2)] \ cyclical(:,1);
a2 = a(2);

% Estimate c1 and c2 using IV, without a constant
Z = cyclical(:,1:2); % instruments
Xiv = u(:,1:2);      % endogenous regressors

c = (Z' * Xiv) \ (Z' * u(:,3));

c1 = c(1);
c2 = c(2);

% Rotation matrix

R = zeros(N,N);

den = 1 - c1*a1 - c2*b1;

R(3,1) = (c1 + c2*b2) / den;
R(3,2) = (c1*a2 + c2) / den;
R(3,3) = 1 / den;

R(1,1) = a1*R(3,1) + 1;
R(1,2) = a1*R(3,2) + a2;
R(1,3) = a1*R(3,3);

R(2,1) = b1*R(3,1) + b2;
R(2,2) = b1*R(3,2) + 1;
R(2,3) = b1*R(3,3);

% Impulse response functions

S = priors.horizon;

THETA = cell(S,1);
STHETA = cell(S,1);

THETA{1} = eye(N);
STHETA{1} = THETA{1} * R;

for s = 2:S

    THETA{s} = zeros(N);

    for k = 1:P
        if s-k > 0
            THETA{s} = THETA{s} + PHI{k}*THETA{s-k};
        end
    end

    STHETA{s} = THETA{s} * R;
end

% Preallocate IRFs
IRF_TG = zeros(S,1);
IRF_GG = zeros(S,1);
IRF_YG = zeros(S,1);

% Responses to structural government-spending shock: column 2
for s = 1:S
    IRF_TG(s) = STHETA{s}(1,2);
    IRF_GG(s) = STHETA{s}(2,2);
    IRF_YG(s) = STHETA{s}(3,2);
end

% Scale to dollar responses to a one-dollar spending shock
IRF_TGsca = IRF_TG ./ gtot;
IRF_GGsca = IRF_GG;
IRF_YGsca = IRF_YG ./ gtoy;

% S x 3 matrix: responses of taxes, spending and GDP
IRF_G = [IRF_TGsca, IRF_GGsca, IRF_YGsca];

end