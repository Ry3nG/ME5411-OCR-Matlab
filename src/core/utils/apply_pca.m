function [Xtr_pca, Xte_pca, pcaM] = apply_pca(Xtr, Xte, var_keep)
% APPLY_PCA Apply PCA dimensionality reduction (Part-II classical method)
%
% Inputs:
%   Xtr      - [N_train x D] training feature matrix
%   Xte      - [N_test x D] test feature matrix
%   var_keep - Fraction of variance to retain (e.g., 0.95 for 95%)
%
% Outputs:
%   Xtr_pca  - [N_train x k] PCA-transformed training features
%   Xte_pca  - [N_test x k] PCA-transformed test features
%   pcaM     - PCA model struct containing:
%              .mu: mean vector
%              .W: projection matrix [D x k]
%              .k: number of components retained
%              .var_keep: variance retention ratio
%              .explained_var: cumulative explained variance
%
% Theory:
%   PCA finds the k principal components that capture var_keep (e.g., 95%)
%   of the total variance. This is a classical Part-II dimensionality reduction
%   technique that can improve SVM training speed and sometimes accuracy.

    % 1) Center the training data
    mu = mean(Xtr, 1);                    % [1 x D]
    Xc = bsxfun(@minus, Xtr, mu);         % [N x D]

    % 2) Compute covariance matrix
    C = (Xc' * Xc) / (size(Xtr, 1) - 1);  % [D x D]

    % 3) Eigen decomposition
    [V, S] = eig(C);
    [s, ord] = sort(diag(S), 'descend');  % Sort eigenvalues in descending order
    V = V(:, ord);                        % Reorder eigenvectors

    % 4) Determine number of components k to retain var_keep variance
    cumsum_var = cumsum(s) / sum(s);
    k = find(cumsum_var >= var_keep, 1);
    if isempty(k)
        k = length(s);  % Use all components if threshold not met
    end

    % 5) Projection matrix (top k eigenvectors)
    W = V(:, 1:k);                        % [D x k]

    % 6) Project training and test data
    Xtr_pca = (Xtr - mu) * W;             % [N_train x k]
    Xte_pca = (Xte - mu) * W;             % [N_test x k]

    % 7) Package PCA model
    pcaM = struct();
    pcaM.mu = mu;
    pcaM.W = W;
    pcaM.k = k;
    pcaM.var_keep = var_keep;
    pcaM.explained_var = cumsum_var(k);

    fprintf('PCA: Reduced %d → %d dimensions (%.2f%% variance retained)\n', ...
            size(Xtr, 2), k, pcaM.explained_var * 100);
end
