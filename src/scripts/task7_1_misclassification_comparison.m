% Task 7.1: Misclassification Feature Map Comparison
% Compare feature maps between correctly classified and misclassified samples
% to understand why the network confuses certain characters

clear all; %#ok<CLALL>
close all;

% Get the project root directory
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

% Add paths
addpath(genpath('src/core'));
addpath(genpath('src/utils'));

% Load trained model and predictions
result_dir = 'output/task7_1/11-03_03-17-04/';
fprintf('Loading model from: %s\n', result_dir);
load(fullfile(result_dir, 'cnn_best_acc.mat'), 'cnn');
load(fullfile(result_dir, 'predictions.mat'), 'preds', 'labels_test_eval');
load('data/test.mat', 'data_test', 'labels_test');

% Convert labels
labels_test = labels_test + 1;

class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};

% Output directory
output_dir = 'output/task7_1/11-03_03-17-04/figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Find confusion pairs
% Based on confusion matrix analysis, top confusions:
% D->4 (22 cases), H->A (18 cases), 8->A (12 cases), 8->H (8 cases)

% Let's visualize: D->4 confusion (most common)
% Find one case where D was misclassified as 4
true_class = 5;   % D (0-indexed: 5, class_names index: 6)
pred_class = 1;   % 4 (0-indexed: 1, class_names index: 2)

misclass_idx = find((labels_test_eval == true_class) & (preds == pred_class), 1);
if isempty(misclass_idx)
    fprintf('Warning: No D->4 misclassification found, trying alternative...\n');
    % Try H->A
    true_class = 6; pred_class = 4; % H->A
    misclass_idx = find((labels_test_eval == true_class) & (preds == pred_class), 1);
    if isempty(misclass_idx)
        error('No misclassification examples found');
    end
end

% Find a correctly classified sample from the predicted class (4)
correct_idx = find((labels_test_eval == pred_class) & (preds == pred_class), 1);

fprintf('Misclassified: True=%s, Pred=%s, Index=%d\n', ...
    class_names{true_class+1}, class_names{pred_class+1}, misclass_idx);
fprintf('Correct: Class=%s, Index=%d\n', class_names{pred_class+1}, correct_idx);

%% Extract feature maps for both samples
local_option.train_mode = false;

% Misclassified sample
sample_mis = data_test(:,:,1,misclass_idx);
cnn_mis = cnn;
cnn_mis = forward(cnn_mis, sample_mis, local_option);
conv3_mis = cnn_mis.layers{4}.activations;  % 12x12x64

% Correctly classified sample
sample_cor = data_test(:,:,1,correct_idx);
cnn_cor = cnn;
cnn_cor = forward(cnn_cor, sample_cor, local_option);
conv3_cor = cnn_cor.layers{4}.activations;  % 12x12x64

%% Select top 16 most activated filters for visualization
% For misclassified sample
conv3_mis_strength = squeeze(sum(sum(abs(conv3_mis), 1), 2));
[~, top_idx_mis] = sort(conv3_mis_strength, 'descend');
top_16_mis = top_idx_mis(1:16);

% For correct sample
conv3_cor_strength = squeeze(sum(sum(abs(conv3_cor), 1), 2));
[~, top_idx_cor] = sort(conv3_cor_strength, 'descend');
top_16_cor = top_idx_cor(1:16);

%% Generate comparison figure
fprintf('Generating misclassification comparison...\n');

fig = figure('Position', [100, 100, 1400, 600], 'Color', 'white');

% Row 1: Misclassified D->4
subplot(2, 9, 1);
imshow(sample_mis, []);
axis off;
title(sprintf('True: %s\nPred: %s', class_names{true_class+1}, class_names{pred_class+1}), ...
    'FontSize', 10, 'Color', 'red', 'FontWeight', 'bold');

% Show top 16 Conv3 feature maps (4x2 layout in remaining 8 columns)
for i = 1:8
    subplot(2, 9, i+1);
    filter_idx = top_16_mis(i);
    imshow(conv3_mis(:,:,filter_idx), []);
    axis off;
end

% Row 2: Correctly classified 4
subplot(2, 9, 10);
imshow(sample_cor, []);
axis off;
title(sprintf('True: %s\nPred: %s', class_names{pred_class+1}, class_names{pred_class+1}), ...
    'FontSize', 10, 'Color', 'green', 'FontWeight', 'bold');

for i = 1:8
    subplot(2, 9, i+10);
    filter_idx = top_16_cor(i);
    imshow(conv3_cor(:,:,filter_idx), []);
    axis off;
end

set(gcf, 'Color', 'white');
saveas(fig, fullfile(output_dir, 'misclassification_comparison.png'));
close(fig);

%% Generate individual images for LaTeX subfigure layout

% Misclassified input
fig = figure('Position', [100, 100, 300, 300], 'Color', 'white');
imshow(sample_mis, []);
axis off;
set(gca, 'Position', [0 0 1 1]);
saveas(fig, fullfile(output_dir, 'misclass_input.png'));
close(fig);

% Misclassified Conv3 (top 16 in 4x4 grid)
fig = figure('Position', [100, 100, 800, 800], 'Color', 'white');
for i = 1:16
    subplot(4, 4, i);
    filter_idx = top_16_mis(i);
    imshow(conv3_mis(:,:,filter_idx), []);
    axis off;
end
set(gcf, 'Color', 'white');
saveas(fig, fullfile(output_dir, 'misclass_conv3.png'));
close(fig);

% Correct input
fig = figure('Position', [100, 100, 300, 300], 'Color', 'white');
imshow(sample_cor, []);
axis off;
set(gca, 'Position', [0 0 1 1]);
saveas(fig, fullfile(output_dir, 'correct_input.png'));
close(fig);

% Correct Conv3 (top 16 in 4x4 grid)
fig = figure('Position', [100, 100, 800, 800], 'Color', 'white');
for i = 1:16
    subplot(4, 4, i);
    filter_idx = top_16_cor(i);
    imshow(conv3_cor(:,:,filter_idx), []);
    axis off;
end
set(gcf, 'Color', 'white');
saveas(fig, fullfile(output_dir, 'correct_conv3.png'));
close(fig);

%% Summary
fprintf('\n=== Misclassification Comparison Complete ===\n');
fprintf('Comparison: %s (misclassified as %s) vs %s (correct)\n', ...
    class_names{true_class+1}, class_names{pred_class+1}, class_names{pred_class+1});
fprintf('Files generated:\n');
fprintf('  - misclassification_comparison.png (full comparison)\n');
fprintf('  - misclass_input.png (misclassified input)\n');
fprintf('  - misclass_conv3.png (misclassified Conv3 features)\n');
fprintf('  - correct_input.png (correct input)\n');
fprintf('  - correct_conv3.png (correct Conv3 features)\n');
fprintf('\nThese show why the network confuses visually similar characters.\n');
