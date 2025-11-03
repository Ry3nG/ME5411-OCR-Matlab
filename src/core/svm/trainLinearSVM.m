function model = trainLinearSVM(X, y, C, varargin)
% TRAINLINEARSVM Train a linear SVM using SGD with hinge loss
%
% Inputs:
%   X        - [N x D] feature matrix
%   y        - [N x 1] binary labels (+1 or -1)
%   C        - Regularization parameter (smaller = more regularization)
%   varargin - Optional parameters:
%              'max_epochs' (default: 100)
%              'lr' (default: 0.01)
%              'verbose' (default: false)
%
% Outputs:
%   model - Struct containing:
%           .w: weight vector
%           .b: bias term

    % Parse optional parameters
    p = inputParser;
    addParameter(p, 'max_epochs', 100);
    addParameter(p, 'lr', 0.01);
    addParameter(p, 'verbose', false);
    parse(p, varargin{:});

    max_epochs = p.Results.max_epochs;
    lr = p.Results.lr;
    verbose = p.Results.verbose;

    [N, D] = size(X);
    X = [X, ones(N, 1)];      % Feature augmentation to absorb bias

    % Initialize parameters
    w = randn(D + 1, 1) * 0.01;  % Small random initialization

    % Regularization weight
    lambda = 1 / (C * N);  % Regularization strength

    % SGD training
    for epoch = 1:max_epochs
        % Shuffle data
        indices = randperm(N);

        % One epoch through all data
        for i = 1:N
            idx = indices(i);
            xi = X(idx, :)';
            yi = y(idx);

            % Compute margin
            margin = yi * (w' * xi);

            % Subgradient update
            if margin < 1
                % Misclassified or within margin: update both w and b
                w = w - lr * (lambda * w - yi * xi);
            else
                % Correctly classified outside margin: only regularize w
                w = w - lr * lambda * w;
            end
        end

        % Progress display
        if verbose && mod(epoch, max_epochs/10) == 0
            % Compute training accuracy
            pred = sign(X * w);
            pred(pred == 0) = 1;  % Handle zero predictions
            acc = sum(pred == y) / N;
            fprintf('    Epoch %d/%d, Acc: %.2f%%\n', epoch, max_epochs, acc*100);
        end
    end

    % Package model
    model.w = w(1:end-1);
    model.b = w(end);
end
