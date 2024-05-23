function x = findQP_LinearConstraints(P, q, A, b, maxIterations)
% This function solves a Quadratic program with Linear Constraints
% minimize (x' * P * x + q' * x)
% subject to: A * x <= b
% The function uses interior point method to solve the problem
% We start from a feasible point. For this code, we start by zero as x0

xDimension = length(P); % dimension of x
constraintDimension = size(A,1); % number of constraints
%MAXITERS = 5;
TOL = 1e-6;
residualTol = 1e-8; % Tolerance on residuals
MU = 10;
ALPHA = 0.01;
BETA = 0.5;
x = zeros(xDimension,1); % We assume zero is always a feasible point 
primalResidual = b - A * x; 
primalResidual = primalResidual + 1e-6; % Add by a small number to prevent division by zero
z = 1./primalResidual;
counter = 0;
for iters = 1:maxIterations
    
    centralityResidual = primalResidual'*z; 

    dualResidual = P*x + q + A' * z ; % Gradient of the Lagrangian
    

    if ((centralityResidual < TOL) && (norm(dualResidual) < residualTol)) 
        break; 
    end
    
    %% Find Search Direction
    % We apply a Newton step to find the search direction

    tinv = centralityResidual/(constraintDimension * MU);
    HessianMatrix = -[ P A'; A diag(-primalResidual./z) ];
    residualVector = [ dualResidual; -primalResidual + tinv*(1./z) ];
    searchDirection = HessianMatrix \ residualVector;
    dx = searchDirection(1:xDimension); 
    dz = searchDirection(xDimension+[1:constraintDimension]); 
    ds = -A*dx;
    r = [P*x+q+A'*z; z.*primalResidual-tinv];
    %step = min(1.0, 0.99/max(-dz./z));
    
    %% Line Search for the search step
    step = 1;
    while (min(primalResidual+step*ds) <= 0)
        step = BETA*step; 
    end

    %% Update Primal-Dual Variables
    newz = z+step*dz; 
    newx = x+step*dx; 
    news = primalResidual+step*ds;
    newResidualVector = [P*newx+q+A'*newz; newz.*news-tinv];
    
    while (norm(newResidualVector) > (1-ALPHA*step)*norm(r))
        step = BETA*step;
        newz = z+step*dz; newx = x+step*dx; news = primalResidual+step*ds;
        newResidualVector = [P*newx+q+A'*newz; newz.*news-tinv];
    end
    
    x = x+step*dx; 
    z = z +step*dz; 
    primalResidual = b - A * x;
    counter = counter + 1;
    centralityGap(counter) = norm(centralityResidual);
    primalGap(counter) = norm(primalResidual);
end
primalGap
subplot(2,1,1)
plot(1:counter, primalGap, 'LineWidth',2)
grid
xlabel ("Iterations")
xlabel ("Primal Gap")

subplot(2,1,2)
plot(1:counter, centralityGap, 'LineWidth',2)
grid
xlabel ("Iterations")
xlabel ("Centrality Gap")
