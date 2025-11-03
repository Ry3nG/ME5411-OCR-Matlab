function [margins, top2] = multiclass_margins(models, X, y)
% MULTICLASS_MARGINS Compute margin between true class and second-best class
%
% Inputs:
%   models - Cell array of binary SVM models
%   X      - [N x D] feature matrix
%   y      - [N x 1] true class labels (1 to C)
%
% Outputs:
%   margins - [N x 1] margin (true_score - second_best_score)
%   top2    - [N x 1] second-best predicted class
%
% Theory: SVM margin represents confidence. Small margins indicate
%         hard examples near decision boundary (support vectors).

    C = numel(models);
    N = size(X, 1);

    % Compute decision scores for all classes
    S = zeros(N, C);
    for c = 1:C
        S(:, c) = X * models{c}.w + models{c}.b;
    end

    % Sort scores to find top-1 and top-2
    [v, idx] = sort(S, 2, 'descend');

    % Compute margin: true_class_score - second_best_score
    margins = zeros(N, 1);
    top2 = zeros(N, 1);
    for i = 1:N
        true_score = S(i, y(i));
        % Remove true class from sorted scores to find second-best
        scores_without_true = S(i, :);
        scores_without_true(y(i)) = -inf;
        [second_score, second_idx] = max(scores_without_true);
        margins(i) = true_score - second_score;
        top2(i) = second_idx;
    end
end
