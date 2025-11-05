% Check baseline predictions
load('output/task8_2_preprocess/baseline/results.mat');
fprintf('Ground truth: 7 H D 4 4 7 8 0 A 0 0\n');
fprintf('Predictions:  ');
class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
for i = 1:length(results.real_predictions)
    fprintf('%s ', class_names{results.real_predictions(i)});
end
fprintf('\n');
