function Xn = normalize_rows(X)
% NORMALIZE_ROWS Normalize each row to zero mean and unit variance
%
% Inputs:
%   X - [N x D] matrix where each row is a sample
%
% Outputs:
%   Xn - [N x D] normalized matrix
%
% Theory: Zero-mean unit-variance normalization makes features invariant
%         to affine transformations, improving SOM clustering quality.

    mu = mean(X, 2);
    sd = std(X, 0, 2);
    sd(sd < 1e-6) = 1;           % Prevent division by zero
    Xn = (X - mu) ./ sd;
end
