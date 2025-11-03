% Task 7.1: Feature Map Evolution Visualization
% Visualize feature maps from all 3 convolutional layers (Conv1, Conv2, Conv3)
% to show hierarchical feature learning

clear all; %#ok<CLALL>
close all;

% Get the project root directory
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

% Add paths
addpath(genpath('src/core'));
addpath(genpath('src/utils'));

% Load trained model
result_dir = 'output/task7_1/11-03_03-17-04/';
fprintf('Loading model from: %s\n', result_dir);
load(fullfile(result_dir, 'cnn_best_acc.mat'), 'cnn');
load('data/test.mat', 'data_test', 'labels_test');

% Convert labels
labels_test = labels_test + 1;

class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};

% Output directory for feature evolution figures
output_dir = 'output/task7_1/11-03_03-17-04/figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Select one sample from digit '8' (class index 3 in 0-indexed)
preds_temp = predict(cnn, data_test);
preds_temp = preds_temp - 1;
labels_test_eval = labels_test - 1;

correct_idx = find((labels_test_eval == 3) & (preds_temp == 3), 1);  % Class 8
if isempty(correct_idx)
    error('No correctly classified sample found for digit 8');
end

sample_img = data_test(:,:,1,correct_idx);
fprintf('Selected sample: index %d (digit 8)\n', correct_idx);

%% Forward pass to extract all layer activations
cnn_temp = cnn;
local_option.train_mode = false;
cnn_temp = forward(cnn_temp, sample_img, local_option);

%% Extract feature maps from each convolutional layer
% Network structure: Input -> Conv1 -> Conv2 -> Conv3 -> FC -> Output
% Layers: {1: input, 2: Conv1, 3: Conv2, 4: Conv3, 5: FC, 6: output}

conv1_activations = cnn_temp.layers{2}.activations;  % 60x60x16
conv2_activations = cnn_temp.layers{3}.activations;  % 28x28x32
conv3_activations = cnn_temp.layers{4}.activations;  % 12x12x64

fprintf('Conv1 activations: %dx%dx%d\n', size(conv1_activations));
fprintf('Conv2 activations: %dx%dx%d\n', size(conv2_activations));
fprintf('Conv3 activations: %dx%dx%d\n', size(conv3_activations));

%% Visualize Conv1 (16 filters) - save individually for LaTeX subfigure
fprintf('Generating Conv1 feature maps...\n');

% Input image
fig = figure('Position', [100, 100, 300, 300], 'Color', 'white');
imshow(sample_img, []);
axis off;
set(gca, 'Position', [0 0 1 1]);
saveas(fig, fullfile(output_dir, 'feature_input.png'));
close(fig);

% Conv1 feature maps (16 filters in 4x4 grid)
fig = figure('Position', [100, 100, 800, 800], 'Color', 'white');
for i = 1:16
    subplot(4, 4, i);
    imshow(conv1_activations(:,:,i), []);
    axis off;
end
set(gcf, 'Color', 'white');
saveas(fig, fullfile(output_dir, 'feature_conv1.png'));
close(fig);

%% Visualize Conv2 (32 filters in 6x6 grid)
fprintf('Generating Conv2 feature maps...\n');

fig = figure('Position', [100, 100, 1200, 1200], 'Color', 'white');
for i = 1:32
    subplot(6, 6, i);
    imshow(conv2_activations(:,:,i), []);
    axis off;
end
set(gcf, 'Color', 'white');
saveas(fig, fullfile(output_dir, 'feature_conv2.png'));
close(fig);

%% Visualize Conv3 (64 filters in 8x8 grid)
fprintf('Generating Conv3 feature maps...\n');

fig = figure('Position', [100, 100, 1600, 1600], 'Color', 'white');
for i = 1:64
    subplot(8, 8, i);
    imshow(conv3_activations(:,:,i), []);
    axis off;
end
set(gcf, 'Color', 'white');
saveas(fig, fullfile(output_dir, 'feature_conv3.png'));
close(fig);

%% Summary
fprintf('\n=== Feature Evolution Visualization Complete ===\n');
fprintf('Saved to: %s\n', output_dir);
fprintf('Files generated:\n');
fprintf('  - feature_input.png (input image)\n');
fprintf('  - feature_conv1.png (16 filters, 4x4 grid)\n');
fprintf('  - feature_conv2.png (32 filters, 6x6 grid)\n');
fprintf('  - feature_conv3.png (64 filters, 8x8 grid)\n');
fprintf('\nThese images can be combined in LaTeX using subfigure environment.\n');
