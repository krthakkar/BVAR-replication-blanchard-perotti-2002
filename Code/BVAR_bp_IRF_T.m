function [IRF_T] = BVAR_bp_IRF_T(draw,priors,data)

Y = data.Y; % T x N
X = data.X; % T x (ndt terms + N*lag)

N = priors.N;
T = data.T;
P = priors.Lag;
ndt = priors.ndt;

EGY = data.EGY;
ETY = data.ETY;

ttoy = data.ttoy;
ttog = data.ttog;


% Extract deterministic and lag coefficients
PHI0 = draw.phivar(:,1:ndt);

PHI = cell(P,1);
for k = 1:P
   PHI{k} = draw.phivar(:,ndt+1+(k-1)*N:ndt+k*N); 
end

% Reduced-form residuals
u = Y - X * draw.phivar';

% u is T x N

% B&P (2002) identification

% Taxes ordered first
a1 = 2.08;
b1 = 0;
a2 = 0;

% Exact BP spending adjustment because b1 = 0
cyclical(:,1) = u(:,1) - ETY .* u(:,3);
cyclical(:,2) = u(:,2);

% Estimate b2
b = [ones(T,1), cyclical(:,1)] \ cyclical(:,2);
b2 = b(2);

% Estimate c1 and c2
Z = cyclical(:,1:2);
Xiv = u(:,1:2);

c = (Z' * Xiv) \ (Z' * u(:,3));

c1 = c(1);
c2 = c(2);

% Tax-first rotation matrix
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
IRF_TT = zeros(S,1);
IRF_GT = zeros(S,1);
IRF_YT = zeros(S,1);

% Shock to government spending
for s = 1:S
    IRF_TT(s) = STHETA{s}(1,1);
    IRF_GT(s) = STHETA{s}(2,1);
    IRF_YT(s) = STHETA{s}(3,1);
end

% Scaled IRFs: government-spending shock
IRF_TTsca = IRF_TT;
IRF_GTsca = IRF_GT ./ ttog;
IRF_YTsca = IRF_YT ./ ttoy;

% S x 3 matrices
IRF_T = [IRF_TTsca, IRF_GTsca, IRF_YTsca];

end

