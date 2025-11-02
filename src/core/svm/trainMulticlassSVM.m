function models = trainMulticlassSVM(X, y, num_classes, C, varargin)
% TRAINMULTICLASSSVM Train multi-class SVM using one-vs-rest strategy
%
% Inputs:
%   X           - [N x D] feature matrix
%   y           - [N x 1] class labels (1 to num_classes)
%   num_classes - Number of classes
%   C           - Regularization parameter
%   varargin    - Optional parameters passed to trainLinearSVM
%
% Outputs:
%   models - Cell array of binary SVM models

    fprintf('Training multi-class SVM (%d classes) using one-vs-rest...\n', num_classes);

    models = cell(num_classes, 1);

    for c = 1:num_classes
        fprintf('  Training classifier for class %d...\n', c);

        % Create binary labels: +1 for class c, -1 for others
        y_binary = ones(size(y));
        y_binary(y ~= c) = -1;

        % Train binary SVM
        models{c} = trainLinearSVM(X, y_binary, C, varargin{:});
    end

    fprintf('Multi-class SVM training completed.\n');
end
