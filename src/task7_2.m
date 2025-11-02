%% Task 7.2: Non-CNN Character Classification using SOM + Bag-of-Visual-Words + SVM
% ME5411 OCR Project
% Method: Self-Organizing Maps for visual codebook learning,
%         Bag-of-Visual-Words for feature encoding,
%         Support Vector Machines for classification
%
% This approach uses only methods covered in Part 2 of the course:
% - SOM: Unsupervised learning for prototype discovery
% - SVM: Supervised classification

clear; clc; close all;

%% Get the project root directory
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

%% Setup paths
addpath(genpath('src/core'));
output_dir = 'output/task7_2';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load preprocessed data
fprintf('=== Task 7.2: SOM + BoW + SVM Classification ===\n\n');
fprintf('Loading data...\n');
load('data/train.mat', 'data_train', 'labels_train');
load('data/test.mat', 'data_test', 'labels_test');

% Rename for consistency
trainData = data_train;
trainLabels = labels_train + 1;  % Convert from 0-6 to 1-7
testData = data_test;
testLabels = labels_test + 1;    % Convert from 0-6 to 1-7
clear data_train labels_train data_test labels_test;

% Get dimensions
[H, W, ~, N_train] = size(trainData);
N_test = size(testData, 4);
num_classes = 7;

% Class names (mapping from 1-7 to actual characters)
class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};

fprintf('Training samples: %d\n', N_train);
fprintf('Test samples: %d\n', N_test);
fprintf('Image size: %dx%d\n', H, W);
fprintf('Number of classes: %d\n', num_classes);
fprintf('Class names: %s\n\n', strjoin(class_names, ', '));

%% Hyperparameters
% Patch extraction
patch_size = 8;           % 8x8 patches
num_patch_samples = 100000; % Number of patches for SOM training

% SOM configuration
som_grid_size = [10, 10];   % 10x10 grid = 100 visual words
som_iterations = 25000;     % Training iterations
som_lr_init = 0.5;
som_lr_final = 0.01;

% Feature extraction
stride = 4;               % 50% overlap for dense sampling
norm_type = 'l2';         % L2 normalization for BoW histograms
soft_voting = true;       % Use soft voting with SOM neighborhood
sigma_bow = 0.75;         % Neighborhood sigma for soft voting
spatial_pyramid = true;   % Use 1x1 + 2x2 spatial pyramid

% SVM configuration
C = 1.0;                  % Regularization parameter (higher = less regularization)
svm_max_epochs = 200;     % Number of epochs
svm_lr = 0.1;             % Learning rate

fprintf('=== Hyperparameters ===\n');
fprintf('Patch size: %dx%d\n', patch_size, patch_size);
fprintf('Number of patches for SOM: %d\n', num_patch_samples);
fprintf('SOM grid: %dx%d (%d neurons)\n', som_grid_size(1), som_grid_size(2), prod(som_grid_size));
fprintf('SOM iterations: %d\n', som_iterations);
fprintf('Feature extraction stride: %d\n', stride);
fprintf('Histogram normalization: %s\n', norm_type);
fprintf('Soft voting: %s (sigma=%.2f)\n', string(soft_voting), sigma_bow);
fprintf('Spatial pyramid: %s\n', string(spatial_pyramid));
fprintf('SVM C parameter: %.2f\n\n', C);

%% Stage 1: Extract patches for SOM training
fprintf('=== Stage 1: Patch Extraction ===\n');
tic;
patches = extract_patches(trainData, trainLabels, patch_size, num_patch_samples, ...
    'normalize', true, 'verbose', true);
t_patches = toc;
fprintf('Patch extraction time: %.2f seconds\n\n', t_patches);

%% Stage 2: Train SOM for visual codebook
fprintf('=== Stage 2: SOM Training ===\n');
tic;
som_model = train_som(patches, som_grid_size, som_iterations, ...
    'lr_init', som_lr_init, ...
    'lr_final', som_lr_final, ...
    'verbose', true);
t_som = toc;
fprintf('SOM training time: %.2f seconds\n\n', t_som);

% Save SOM model
save(fullfile(output_dir, 'som_model.mat'), 'som_model');

%% Stage 3: Extract BoW features from training data
fprintf('=== Stage 3: Training Feature Extraction ===\n');
tic;
train_features = extract_bow_features(trainData, som_model, stride, ...
    'normalize', true, ...
    'norm_type', norm_type, ...
    'soft_voting', soft_voting, ...
    'sigma_bow', sigma_bow, ...
    'spatial_pyramid', spatial_pyramid, ...
    'verbose', true);
t_train_features = toc;
fprintf('Training feature extraction time: %.2f seconds\n\n', t_train_features);

%% Stage 4: Train multi-class SVM
fprintf('=== Stage 4: SVM Training ===\n');
tic;
svm_models = trainMulticlassSVM(train_features, trainLabels, num_classes, C, ...
    'max_epochs', svm_max_epochs, ...
    'lr', svm_lr, ...
    'verbose', true);
t_svm = toc;
fprintf('SVM training time: %.2f seconds\n\n', t_svm);

% Save SVM models
save(fullfile(output_dir, 'svm_models.mat'), 'svm_models');

% Total training time
t_total_train = t_patches + t_som + t_train_features + t_svm;
fprintf('=== Training Summary ===\n');
fprintf('Total training time: %.2f seconds (%.2f minutes)\n\n', ...
    t_total_train, t_total_train/60);

%% Stage 5: Extract BoW features from test data
fprintf('=== Stage 5: Test Feature Extraction ===\n');
tic;
test_features = extract_bow_features(testData, som_model, stride, ...
    'normalize', true, ...
    'norm_type', norm_type, ...
    'soft_voting', soft_voting, ...
    'sigma_bow', sigma_bow, ...
    'spatial_pyramid', spatial_pyramid, ...
    'verbose', true);
t_test_features = toc;
fprintf('Test feature extraction time: %.2f seconds\n\n', t_test_features);

%% Stage 6: Evaluate on test set
fprintf('=== Stage 6: Evaluation ===\n');
tic;
predictions = predictMulticlassSVM(svm_models, test_features);
t_predict = toc;

accuracy = sum(predictions == testLabels) / N_test;
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Prediction time: %.2f seconds\n', t_predict);
fprintf('Average time per sample: %.4f seconds\n\n', t_predict / N_test);

%% Compute confusion matrix
conf_matrix = zeros(num_classes, num_classes);
for i = 1:N_test
    true_class = testLabels(i);
    pred_class = predictions(i);
    conf_matrix(true_class, pred_class) = conf_matrix(true_class, pred_class) + 1;
end

% Compute per-class accuracy
class_accuracy = diag(conf_matrix) ./ sum(conf_matrix, 2);
fprintf('=== Per-Class Accuracy ===\n');
for c = 1:num_classes
    fprintf('Class %d (%s): %.2f%%\n', c, class_names{c}, class_accuracy(c) * 100);
end
fprintf('\n');

%% Stage 7: Visualizations
fprintf('=== Stage 7: Generating Visualizations ===\n');

% 1. Visualize SOM codebook (learned visual prototypes)
fprintf('Generating SOM codebook visualization...\n');
fig = figure('Position', [100, 100, 1000, 1000]);
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
for i = 1:som_model.num_neurons
    subplot(som_grid_size(1), som_grid_size(2), i);

    % Reshape weight vector back to patch
    patch = reshape(som_model.weights(i, :), [patch_size, patch_size]);

    % Normalize for visualization
    patch = (patch - min(patch(:))) / (max(patch(:)) - min(patch(:)) + 1e-10);

    imshow(patch, []);
    axis off;
end
saveas(fig, fullfile(output_dir, 'som_codebook.png'));
close(fig);

% 2. Confusion matrix (matching task7_1 style)
fprintf('Generating confusion matrix...\n');
conf_matrix_norm = conf_matrix ./ sum(conf_matrix, 2);

fig = figure('Position', [100, 100, 900, 800]);
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
imagesc(conf_matrix_norm);
colormap(flipud(gray));  % Black for high values, white for low
colorbar;
caxis([0, 1]);

% Add text annotations (count + percentage)
for i = 1:num_classes
    for j = 1:num_classes
        count = conf_matrix(i, j);
        percentage = conf_matrix_norm(i, j) * 100;

        % Choose text color based on background
        if conf_matrix_norm(i, j) > 0.5
            textColor = 'white';
        else
            textColor = 'black';
        end

        text(j, i, sprintf('%d\n(%.1f%%)', count, percentage), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Color', textColor, ...
            'FontSize', 11);
    end
end

xlabel('Predicted Class', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('True Class', 'FontSize', 12, 'FontWeight', 'bold');
xticks(1:num_classes);
xticklabels(class_names);
yticks(1:num_classes);
yticklabels(class_names);
axis square;
grid off;

saveas(fig, fullfile(output_dir, 'confusion_matrix.png'));
close(fig);

% 3. Per-class accuracy bar chart
fprintf('Generating per-class accuracy chart...\n');
fig = figure('Position', [100, 100, 800, 500]);
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
bar(1:num_classes, class_accuracy * 100, 'FaceColor', [0.2, 0.4, 0.6]);
xlabel('Class', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
xticks(1:num_classes);
xticklabels(class_names);
ylim([0, 100]);
grid on;

% Add value labels on bars
for c = 1:num_classes
    text(c, class_accuracy(c) * 100 + 2, sprintf('%.1f%%', class_accuracy(c) * 100), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

saveas(fig, fullfile(output_dir, 'per_class_accuracy.png'));
close(fig);

% 4. Example BoW histograms for each class (individual figures)
fprintf('Generating per-class BoW histograms...\n');
for c = 1:num_classes
    % Find first test sample of this class
    idx = find(testLabels == c, 1);
    if isempty(idx)
        continue;
    end

    if spatial_pyramid
        % Extract global histogram (first num_neurons dimensions)
        histogram_values = test_features(idx, 1:som_model.num_neurons);
    else
        histogram_values = test_features(idx, :);
    end

    fig = figure('Position', [100, 100, 900, 450]);
    set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
    bar(1:length(histogram_values), histogram_values, 'FaceColor', [0.3, 0.5, 0.7]);
    xlabel('Visual Word Index', 'FontSize', 10);
    ylabel('Frequency (Normalized)', 'FontSize', 10);
    xlim([0, length(histogram_values) + 1]);
    grid on;
    saveas(fig, fullfile(output_dir, sprintf('bow_hist_%s.png', class_names{c})));
    close(fig);
end

% Remove legacy collage if it exists to avoid confusion
legacy_hist_path = fullfile(output_dir, 'bow_histograms.png');
if exist(legacy_hist_path, 'file')
    delete(legacy_hist_path);
end

% 5. Misclassification examples
fprintf('Generating misclassification examples...\n');
misclassified_idx = find(predictions ~= testLabels);
num_examples = min(12, length(misclassified_idx));

if num_examples > 0
    fig = figure('Position', [100, 100, 1200, 800]);
    set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
    for i = 1:num_examples
        idx = misclassified_idx(i);
        subplot(3, 4, i);
        imshow(squeeze(testData(:, :, 1, idx)));
        hold on;
        true_class = testLabels(idx);
        pred_class = predictions(idx);
        % Overlay prediction info inside the image to avoid figure captions
        text(4, 12, sprintf('T:%s P:%s', class_names{true_class}, class_names{pred_class}), ...
            'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold', ...
            'BackgroundColor', [0, 0, 0], 'Margin', 1);
        hold off;
    end
    saveas(fig, fullfile(output_dir, 'misclassifications.png'));
    close(fig);
end

fprintf('All visualizations saved.\n\n');

%% Save results
fprintf('=== Saving Results ===\n');
results.accuracy = accuracy;
results.confusion_matrix = conf_matrix;
results.class_accuracy = class_accuracy;
results.predictions = predictions;
results.training_time = t_total_train;
results.test_time = t_test_features + t_predict;
results.hyperparameters = struct(...
    'patch_size', patch_size, ...
    'num_patch_samples', num_patch_samples, ...
    'som_grid_size', som_grid_size, ...
    'som_iterations', som_iterations, ...
    'stride', stride, ...
    'norm_type', norm_type, ...
    'C', C, ...
    'svm_max_epochs', svm_max_epochs);

save(fullfile(output_dir, 'results.mat'), 'results');
fprintf('Results saved to %s\n\n', output_dir);

%% Final summary
fprintf('=== Task 7.2 Complete ===\n');
fprintf('Method: SOM + Bag-of-Visual-Words + SVM\n');
fprintf('Visual Codebook Size: %d\n', som_model.num_neurons);
if spatial_pyramid
    fprintf('Feature Dimension: %d (with 1x1+2x2 spatial pyramid)\n', size(test_features, 2));
else
    fprintf('Feature Dimension: %d\n', som_model.num_neurons);
end
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Total Training Time: %.2f minutes\n', t_total_train/60);
fprintf('Total Test Time: %.2f seconds\n', t_test_features + t_predict);
fprintf('\nAll outputs saved to: %s\n', output_dir);
