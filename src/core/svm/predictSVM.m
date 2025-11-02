function predictions = predictSVM(model, X)
% PREDICTSVM Predict using trained linear SVM
%
% Inputs:
%   model - SVM model from trainLinearSVM
%   X     - [N x D] feature matrix
%
% Outputs:
%   predictions - [N x 1] predicted labels (+1 or -1)

    predictions = sign(X * model.w + model.b);

    % Handle zero predictions (assign to positive class)
    predictions(predictions == 0) = 1;
end
