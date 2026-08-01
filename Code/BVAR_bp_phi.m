function [phi] = BVAR_bp_phi(draw,priors,data)

Y = data.Y; % T x N
X = data.X; % T x (No_coef/N)

N = priors.N;
Lag = priors.Lag;
ndt = priors.ndt;

Qvar = draw.Qvar;

QvarInv = Qvar\eye(N);
H0inv = priors.H0\eye(size(priors.H0, 1)); 

phi_ols = (X'*X)\(X'*Y);
phi_ols = phi_ols(:); % vector form of phi

Q1post = (H0inv + kron(QvarInv, X'*X))\eye(size(H0inv, 1));

phi1post = Q1post * (H0inv * priors.phi0 + kron(QvarInv, X'*X) * phi_ols);

Q1chol = chol(Q1post);

accept = 0;

while accept == 0
    phi1temp = phi1post + (randn(1,size(phi1post,1))*Q1chol)';
    
    % Stationarity Check for Phi
    CB1 = reshape(phi1temp,[],N)';
    phi1temp = CB1(:,1+ndt:N*Lag+ndt); % Do not consider the constant
    F1 = zeros(N * Lag, N * Lag);
    F1(1:N,:) = phi1temp;
    for p = 1:Lag-1
        x_start = N * p + 1;
        x_stop = N * (p + 1);
        y_start = N * (p-1) + 1;
        y_stop = N * p;
        F1(x_start:x_stop,y_start:y_stop) = eye(N);
    end
    ee=max(abs(eig(F1))); % Eigenvalues less than 1 for stationary VAR
    S = ee >= 1;
    if S==0 % VAR stable then consider the draw otherwise redraw
        phi = CB1;
        accept = 1;
    end
end



end