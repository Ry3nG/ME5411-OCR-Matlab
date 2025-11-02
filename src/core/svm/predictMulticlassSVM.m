function predictions = predictMulticlassSVM(models, X)
% PREDICTMULTICLASSSVM Predict using multi-class SVM (one-vs-rest)
%
% Inputs:
%   models - Cell array of binary SVM models
%   X      - [N x D] feature matrix
%
% Outputs:
%   predictions - [N x 1] predicted class labels

    num_classes = length(models);
    N = size(X, 1);

    % Compute decision scores for all classes
    scores = zeros(N, num_classes);
    for c = 1:num_classes
        scores(:, c) = X * models{c}.w + models{c}.b;
    end

    % Predict class with highest score
    [~, predictions] = max(scores, [], 2);
end
