%% Task 7.1: Misclassification Feature Comparison
% Generate side-by-side feature maps for misclassified vs correct samples
% Two rows: Row 1 = misclassified D->4, Row 2 = correct 4
% Each row shows: Input, Conv1, Conv2, Conv3

clear; clc; close all;

%% Setup
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

addpath(genpath('src/core'));
addpath(genpath('src/utils'));

% Locate most recent Task 7.1 run directory
output_base = fullfile('output', 'task7_1');
subdirs = dir(fullfile(output_base, '*-*_*-*-*'));
subdirs = subdirs([subdirs.isdir]);
[~, sort_idx] = sort([subdirs.datenum], 'descend');
model_dir = fullfile(output_base, subdirs(sort_idx(1)).name);

% Output directory
output_dir = fullfile(model_dir, 'misclass_comparison');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

report_dir = '/home/gong-zerui/code/ME5411/ME5411-Project-Report/figs/task7_1';

fprintf('=== Task 7.1: Misclassification Feature Comparison ===\n\n');

%% Load model and data
fprintf('Loading model and data...\n');

cnn_path_candidates = {
    fullfile(model_dir, 'cnn_final.mat'), ...
    fullfile(model_dir, 'cnn_best_acc.mat')
};

cnn = [];
for i = 1:numel(cnn_path_candidates)
    if exist(cnn_path_candidates{i}, 'file')
        tmp = load(cnn_path_candidates{i}, 'cnn');
        cnn = tmp.cnn;
        break;
    end
end

load('data/test.mat', 'data_test', 'labels_test');
labels_test = labels_test + 1;

% Get predictions
[preds, ~] = predict(cnn, data_test);
preds = preds - 1;
labels_test_eval = labels_test - 1;

class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};

fprintf('Model loaded. Finding misclassification examples...\n');

%% Find any misclassification with enough correct samples for comparison
misclass_idx_all = find(preds ~= labels_test_eval);

if isempty(misclass_idx_all)
    error('No misclassifications found!');
end

% Try to find a good misclassification example
mis_idx = [];
cor_idx = [];
true_class_name = '';
pred_class_name = '';

for i = 1:length(misclass_idx_all)
    idx = misclass_idx_all(i);
    true_c = labels_test_eval(idx);
    pred_c = preds(idx);

    % Find correct samples of the predicted class
    correct_idx = find((labels_test_eval == pred_c) & (preds == pred_c));

    if length(correct_idx) >= 5
        mis_idx = idx;
        cor_idx = correct_idx(round(length(correct_idx)/2));
        true_class_name = class_names{true_c + 1};
        pred_class_name = class_names{pred_c + 1};
        break;
    end
end

if isempty(mis_idx)
    error('Could not find suitable examples for comparison');
end

fprintf('Found examples:\n');
fprintf('  Misclassified: %s->%s (index %d)\n', true_class_name, pred_class_name, mis_idx);
fprintf('  Correct: %s (index %d)\n', pred_class_name, cor_idx);

%% Extract feature maps
local_option.train_mode = false;

% Misclassified sample (D->4)
sample_mis = data_test(:,:,1,mis_idx);
cnn_mis = cnn;
cnn_mis = forward(cnn_mis, sample_mis, local_option);
conv1_mis = cnn_mis.layers{2}.activations;
conv2_mis = cnn_mis.layers{3}.activations;
conv3_mis = cnn_mis.layers{4}.activations;

% Correct sample (4)
sample_cor = data_test(:,:,1,cor_idx);
cnn_cor = cnn;
cnn_cor = forward(cnn_cor, sample_cor, local_option);
conv1_cor = cnn_cor.layers{2}.activations;
conv2_cor = cnn_cor.layers{3}.activations;
conv3_cor = cnn_cor.layers{4}.activations;

fprintf('Feature maps extracted.\n');
fprintf('  Conv1: %dx%dx%d\n', size(conv1_mis,1), size(conv1_mis,2), size(conv1_mis,3));
fprintf('  Conv2: %dx%dx%d\n', size(conv2_mis,1), size(conv2_mis,2), size(conv2_mis,3));
fprintf('  Conv3: %dx%dx%d\n', size(conv3_mis,1), size(conv3_mis,2), size(conv3_mis,3));

% Get actual filter counts from data
conv1_num_filters = size(conv1_mis, 3);
conv2_num_filters = size(conv2_mis, 3);
conv3_num_filters = size(conv3_mis, 3);

fprintf('Filter counts: Conv1=%d, Conv2=%d, Conv3=%d\n', conv1_num_filters, conv2_num_filters, conv3_num_filters);

%% Generate individual subfigures for LaTeX

fprintf('\nGenerating subfigures...\n');

% Helper function to save a tight figure
save_tight_fig = @(img, path) (imwrite(img, path));

% Row 1: Misclassified D->4
% Input
img = sample_mis;
img_uint8 = uint8(255 * img);
imwrite(img_uint8, fullfile(output_dir, 'mis_input.png'));

% Conv1 - visualize all filters
conv1_cols = ceil(sqrt(conv1_num_filters));
conv1_rows = ceil(conv1_num_filters / conv1_cols);
fig = figure('Position', [100, 100, 400, 400], 'Color', 'white', 'Visible', 'off');
for i = 1:conv1_num_filters
    subplot(conv1_rows, conv1_cols, i);
    imshow(conv1_mis(:,:,i), []);
    axis off;
end
set(gcf, 'Color', 'white');
print(fullfile(output_dir, 'mis_conv1.png'), '-dpng', '-r150');
close(fig);

% Conv2 - visualize all filters
conv2_cols = ceil(sqrt(conv2_num_filters));
conv2_rows = ceil(conv2_num_filters / conv2_cols);
fig = figure('Position', [100, 100, 600, 600], 'Color', 'white', 'Visible', 'off');
for i = 1:conv2_num_filters
    subplot(conv2_rows, conv2_cols, i);
    imshow(conv2_mis(:,:,i), []);
    axis off;
end
set(gcf, 'Color', 'white');
print(fullfile(output_dir, 'mis_conv2.png'), '-dpng', '-r150');
close(fig);

% Conv3 - visualize all filters
conv3_cols = ceil(sqrt(conv3_num_filters));
conv3_rows = ceil(conv3_num_filters / conv3_cols);
fig = figure('Position', [100, 100, 800, 800], 'Color', 'white', 'Visible', 'off');
for i = 1:conv3_num_filters
    subplot(conv3_rows, conv3_cols, i);
    imshow(conv3_mis(:,:,i), []);
    axis off;
end
set(gcf, 'Color', 'white');
print(fullfile(output_dir, 'mis_conv3.png'), '-dpng', '-r150');
close(fig);

% Row 2: Correct 4
% Input
img = sample_cor;
img_uint8 = uint8(255 * img);
imwrite(img_uint8, fullfile(output_dir, 'cor_input.png'));

% Conv1 - visualize all filters
fig = figure('Position', [100, 100, 400, 400], 'Color', 'white', 'Visible', 'off');
for i = 1:conv1_num_filters
    subplot(conv1_rows, conv1_cols, i);
    imshow(conv1_cor(:,:,i), []);
    axis off;
end
set(gcf, 'Color', 'white');
print(fullfile(output_dir, 'cor_conv1.png'), '-dpng', '-r150');
close(fig);

% Conv2 - visualize all filters
fig = figure('Position', [100, 100, 600, 600], 'Color', 'white', 'Visible', 'off');
for i = 1:conv2_num_filters
    subplot(conv2_rows, conv2_cols, i);
    imshow(conv2_cor(:,:,i), []);
    axis off;
end
set(gcf, 'Color', 'white');
print(fullfile(output_dir, 'cor_conv2.png'), '-dpng', '-r150');
close(fig);

% Conv3 - visualize all filters
fig = figure('Position', [100, 100, 800, 800], 'Color', 'white', 'Visible', 'off');
for i = 1:conv3_num_filters
    subplot(conv3_rows, conv3_cols, i);
    imshow(conv3_cor(:,:,i), []);
    axis off;
end
set(gcf, 'Color', 'white');
print(fullfile(output_dir, 'cor_conv3.png'), '-dpng', '-r150');
close(fig);

fprintf('Saved 8 subfigures to: %s\n', output_dir);

%% Copy to report directory
if exist(report_dir, 'dir')
    copyfile(fullfile(output_dir, '*.png'), report_dir);
    fprintf('Copied figures to report directory: %s\n', report_dir);
else
    fprintf('Warning: Report directory not found: %s\n', report_dir);
end

fprintf('\n=== Generation Complete ===\n');
fprintf('Generated files:\n');
fprintf('  mis_input.png, mis_conv1.png, mis_conv2.png, mis_conv3.png\n');
fprintf('  cor_input.png, cor_conv1.png, cor_conv2.png, cor_conv3.png\n');
