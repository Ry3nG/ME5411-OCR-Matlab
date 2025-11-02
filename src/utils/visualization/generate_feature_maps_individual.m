% Generate Individual Feature Map Images for LaTeX Assembly
% Creates separate PNG files for each sample's feature maps
% Format: horizontal layout (Input | C1 | C3 | C5) similar to the reference example

close all;
clear;

% Add paths
addpath('core/network');
addpath('utils');

% Configuration
log_path = '../output/task7_1/11-02_21-09-20/';
fig_output = log_path + "feature_maps/";
if ~exist(fig_output, 'dir')
    mkdir(fig_output);
end

fprintf('=== Generating Individual Feature Maps ===\n');

%% Load model and data
load([log_path 'cnn_best_acc.mat'], 'cnn');
load('../data/test.mat', 'data_test', 'labels_test');

class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
num_classes = length(class_names);

%% Make predictions
cnn.train_mode = false;
[preds, ~] = predict(cnn, data_test);
preds = preds - 1;

%% Find interesting cases
misclassified_idx = find(preds ~= labels_test);
correct_idx = find(preds == labels_test);

% Select 2-3 misclassified examples (diverse pairs)
fprintf('Selecting diverse misclassification examples...\n');

% Find D->4, H->A, 8->A cases
d_to_4 = misclassified_idx(labels_test(misclassified_idx) == 5 & preds(misclassified_idx) == 1);
h_to_a = misclassified_idx(labels_test(misclassified_idx) == 6 & preds(misclassified_idx) == 4);
eight_to_a = misclassified_idx(labels_test(misclassified_idx) == 3 & preds(misclassified_idx) == 4);

selected_misclass = [];
if ~isempty(d_to_4), selected_misclass(end+1) = d_to_4(1); end
if ~isempty(eight_to_a), selected_misclass(end+1) = eight_to_a(1); end

% Find corresponding correct examples
d_correct = correct_idx(labels_test(correct_idx) == 5);  % D
zero_correct = correct_idx(labels_test(correct_idx) == 0);  % 0

selected_correct = [];
if ~isempty(d_correct), selected_correct(end+1) = d_correct(1); end
if ~isempty(zero_correct), selected_correct(end+1) = zero_correct(1); end

%% Generate feature maps for misclassified examples
fprintf('Generating feature maps for misclassified samples...\n');

for i = 1:length(selected_misclass)
    sample_idx = selected_misclass(i);
    img = data_test(:,:,:,sample_idx);
    true_label = labels_test(sample_idx);
    pred_label = preds(sample_idx);

    % Forward pass
    cnn.train_mode = false;
    cnn = forward(cnn, img, struct('train_mode', false));

    conv1_maps = cnn.layers{2}.activations;  % 30x30x4
    conv2_maps = cnn.layers{3}.activations;  % 13x13x8
    conv3_maps = cnn.layers{4}.activations;  % 3x3x16

    % Create horizontal layout: Input | C1 (4 filters) | C3 (4 top) | C5 (4 top)
    fig = figure('Position', [100, 100, 1400, 300]);

    % Input
    subplot(1, 13, 1);
    imshow(img, []);
    title('Input', 'FontSize', 10);
    ylabel(sprintf('%s\\rightarrow%s', class_names{true_label+1}, class_names{pred_label+1}), ...
        'FontSize', 11, 'FontWeight', 'bold', 'Rotation', 0);

    % Conv1 - all 4 filters
    for f = 1:4
        subplot(1, 13, 1 + f);
        imagesc(conv1_maps(:,:,f));
        colormap(gray);
        axis off;
        title(sprintf('C1-%d', f), 'FontSize', 8);
    end

    % Conv2 - top 4 most active
    conv2_activity = squeeze(max(max(conv2_maps, [], 1), [], 2));
    [~, top_idx] = maxk(conv2_activity, 4);
    for j = 1:4
        f = top_idx(j);
        subplot(1, 13, 5 + j);
        imagesc(conv2_maps(:,:,f));
        colormap(gray);
        axis off;
        title(sprintf('C3-%d', f), 'FontSize', 8);
    end

    % Conv3 - top 4 most active
    conv3_activity = squeeze(max(max(conv3_maps, [], 1), [], 2));
    [~, top_idx] = maxk(conv3_activity, 4);
    for j = 1:4
        f = top_idx(j);
        subplot(1, 13, 9 + j);
        imagesc(conv3_maps(:,:,f));
        colormap(gray);
        axis off;
        title(sprintf('C5-%d', f), 'FontSize', 8);
    end

    % Save with descriptive name
    filename = sprintf('misclass_%s_to_%s.png', ...
        class_names{true_label+1}, class_names{pred_label+1});
    saveas(fig, fullfile(fig_output, filename));
    fprintf('  Saved: %s\n', filename);
    close(fig);
end

%% Generate feature maps for correct examples
fprintf('Generating feature maps for correctly classified samples...\n');

for i = 1:length(selected_correct)
    sample_idx = selected_correct(i);
    img = data_test(:,:,:,sample_idx);
    true_label = labels_test(sample_idx);

    % Forward pass
    cnn.train_mode = false;
    cnn = forward(cnn, img, struct('train_mode', false));

    conv1_maps = cnn.layers{2}.activations;
    conv2_maps = cnn.layers{3}.activations;
    conv3_maps = cnn.layers{4}.activations;

    % Same horizontal layout
    fig = figure('Position', [100, 100, 1400, 300]);

    % Input
    subplot(1, 13, 1);
    imshow(img, []);
    title('Input', 'FontSize', 10);
    ylabel(sprintf('%s', class_names{true_label+1}), ...
        'FontSize', 11, 'FontWeight', 'bold', 'Rotation', 0);

    % Conv1
    for f = 1:4
        subplot(1, 13, 1 + f);
        imagesc(conv1_maps(:,:,f));
        colormap(gray);
        axis off;
        title(sprintf('C1-%d', f), 'FontSize', 8);
    end

    % Conv2
    conv2_activity = squeeze(max(max(conv2_maps, [], 1), [], 2));
    [~, top_idx] = maxk(conv2_activity, 4);
    for j = 1:4
        f = top_idx(j);
        subplot(1, 13, 5 + j);
        imagesc(conv2_maps(:,:,f));
        colormap(gray);
        axis off;
        title(sprintf('C3-%d', f), 'FontSize', 8);
    end

    % Conv3
    conv3_activity = squeeze(max(max(conv3_maps, [], 1), [], 2));
    [~, top_idx] = maxk(conv3_activity, 4);
    for j = 1:4
        f = top_idx(j);
        subplot(1, 13, 9 + j);
        imagesc(conv3_maps(:,:,f));
        colormap(gray);
        axis off;
        title(sprintf('C5-%d', f), 'FontSize', 8);
    end

    % Save
    filename = sprintf('correct_%s.png', class_names{true_label+1});
    saveas(fig, fullfile(fig_output, filename));
    fprintf('  Saved: %s\n', filename);
    close(fig);
end

fprintf('\n=== Feature Map Generation Complete ===\n');
fprintf('Files saved to: %s\n', fig_output);
fprintf('Use these individual images in LaTeX for assembly\n');
