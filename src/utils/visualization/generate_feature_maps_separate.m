% Generate Separate Feature Map Images for LaTeX Assembly
% Creates 4 images per sample: input, c1(2x2), c3(2x2), c5(4x4)

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

fprintf('=== Generating Separate Feature Map Images ===\n');

%% Load model and data
load([log_path 'cnn_best_acc.mat'], 'cnn');
load('../data/test.mat', 'data_test', 'labels_test');

class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};

%% Make predictions
cnn.train_mode = false;
[preds, ~] = predict(cnn, data_test);
preds = preds - 1;

%% Find examples
misclassified_idx = find(preds ~= labels_test);
correct_idx = find(preds == labels_test);

% Select diverse cases
d_to_4 = misclassified_idx(labels_test(misclassified_idx) == 5 & preds(misclassified_idx) == 1);
eight_to_a = misclassified_idx(labels_test(misclassified_idx) == 3 & preds(misclassified_idx) == 4);
d_correct = correct_idx(labels_test(correct_idx) == 5);
zero_correct = correct_idx(labels_test(correct_idx) == 0);

% List of samples to process
samples = [];
if ~isempty(d_to_4), samples(end+1).idx = d_to_4(1); samples(end).type = 'misclass'; end
if ~isempty(eight_to_a), samples(end+1).idx = eight_to_a(1); samples(end).type = 'misclass'; end
if ~isempty(d_correct), samples(end+1).idx = d_correct(1); samples(end).type = 'correct'; end
if ~isempty(zero_correct), samples(end+1).idx = zero_correct(1); samples(end).type = 'correct'; end

%% Process each sample
for s = 1:length(samples)
    sample_idx = samples(s).idx;
    img = data_test(:,:,:,sample_idx);
    true_label = labels_test(sample_idx);
    pred_label = preds(sample_idx);

    % Create prefix for filenames
    if strcmp(samples(s).type, 'misclass')
        prefix = sprintf('misclass_%s_to_%s', ...
            class_names{true_label+1}, class_names{pred_label+1});
        label_str = sprintf('%s→%s', class_names{true_label+1}, class_names{pred_label+1});
    else
        prefix = sprintf('correct_%s', class_names{true_label+1});
        label_str = class_names{true_label+1};
    end

    fprintf('\nProcessing: %s\n', prefix);

    % Forward pass
    cnn.train_mode = false;
    cnn = forward(cnn, img, struct('train_mode', false));

    conv1_maps = cnn.layers{2}.activations;  % 30x30x4
    conv2_maps = cnn.layers{3}.activations;  % 13x13x8
    conv3_maps = cnn.layers{4}.activations;  % 3x3x16

    %% 1. Input image
    fig = figure('Position', [100, 100, 200, 200]);
    imshow(img, []);
    axis off;
    saveas(fig, fullfile(fig_output, [prefix '_input.png']));
    fprintf('  Saved: %s_input.png\n', prefix);
    close(fig);

    %% 2. Conv1: 2x2 grid (4 filters)
    fig = figure('Position', [100, 100, 400, 400]);
    for f = 1:4
        subplot(2, 2, f);
        imagesc(conv1_maps(:,:,f));
        colormap(gray);
        axis off;
    end
    saveas(fig, fullfile(fig_output, [prefix '_c1.png']));
    fprintf('  Saved: %s_c1.png\n', prefix);
    close(fig);

    %% 3. Conv2: 2x4 grid (all 8 filters)
    fig = figure('Position', [100, 100, 800, 400]);
    for f = 1:8
        subplot(2, 4, f);
        imagesc(conv2_maps(:,:,f));
        colormap(gray);
        axis off;
    end
    saveas(fig, fullfile(fig_output, [prefix '_c3.png']));
    fprintf('  Saved: %s_c3.png\n', prefix);
    close(fig);

    %% 4. Conv3: 4x4 grid (all 16 filters)
    fig = figure('Position', [100, 100, 600, 600]);
    for f = 1:16
        subplot(4, 4, f);
        imagesc(conv3_maps(:,:,f));
        colormap(gray);
        axis off;
    end
    saveas(fig, fullfile(fig_output, [prefix '_c5.png']));
    fprintf('  Saved: %s_c5.png\n', prefix);
    close(fig);
end

fprintf('\n=== Generation Complete ===\n');
fprintf('Files saved to: %s\n', fig_output);
fprintf('\nFor each sample, 4 images are generated:\n');
fprintf('  *_input.png - Input image\n');
fprintf('  *_c1.png    - Conv1 (2x2 grid, 4 filters)\n');
fprintf('  *_c3.png    - Conv2 (2x2 grid, top 4 of 8)\n');
fprintf('  *_c5.png    - Conv3 (4x4 grid, all 16 filters)\n');
fprintf('\nUse in LaTeX with horizontal alignment (e.g., subfigure)\n');
