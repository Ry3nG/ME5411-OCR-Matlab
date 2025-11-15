





















































% Debug script to compare training data formats
% Check if train.mat and augmented datasets have different preprocessing

clear; close all; clc;

% Change to project root
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

fprintf('Analyzing data format differences...\n\n');

%% Load baseline training data
baseline = load('data/train.mat');
fprintf('=== Baseline (train.mat) ===\n');
fprintf('  Shape: %s\n', mat2str(size(baseline.data_train)));
fprintf('  Data type: %s\n', class(baseline.data_train));
fprintf('  Value range: [%.6f, %.6f]\n', min(baseline.data_train(:)), max(baseline.data_train(:)));
fprintf('  Unique values (first 100): %d\n', length(unique(baseline.data_train(:, :, 1, 1:min(100, end)))));

% Check if binary (only 0 and 1)
unique_vals = unique(baseline.data_train(:));
is_binary_baseline = length(unique_vals) == 2 && all(ismember(unique_vals, [0, 1]));
fprintf('  Is binary (0/1 only)? %s\n', mat2str(is_binary_baseline));
fprintf('  Mean pixel value: %.6f\n\n', mean(baseline.data_train(:)));

%% Load noise augmented data
if exist('data/train_noise.mat', 'file')
    noise = load('data/train_noise.mat');
    fprintf('=== Noise Augmented (train_noise.mat) ===\n');
    fprintf('  Shape: %s\n', mat2str(size(noise.data_train)));
    fprintf('  Data type: %s\n', class(noise.data_train));
    fprintf('  Value range: [%.6f, %.6f]\n', min(noise.data_train(:)), max(noise.data_train(:)));
    fprintf('  Unique values (first 100): %d\n', length(unique(noise.data_train(:, :, 1, 1:min(100, end)))));

    unique_vals_noise = unique(noise.data_train(:));
    is_binary_noise = length(unique_vals_noise) == 2 && all(ismember(unique_vals_noise, [0, 1]));
    fprintf('  Is binary (0/1 only)? %s\n', mat2str(is_binary_noise));
    fprintf('  Mean pixel value: %.6f\n\n', mean(noise.data_train(:)));
end

%% Visualize sample comparison
fig = figure('Position', [100, 100, 1200, 400], 'Color', 'w');

% Sample 1: baseline
sample_idx = 1;
subplot(2, 4, 1);
img_baseline = baseline.data_train(:, :, 1, sample_idx);
imshow(img_baseline, []);
title(sprintf('Baseline\nClass: %s', num2str(baseline.labels_train(sample_idx))));

% Histogram of baseline
subplot(2, 4, 5);
histogram(img_baseline(:), 50);
title('Baseline Histogram');
xlabel('Pixel Value');
ylabel('Count');
xlim([0, 1]);

% Sample 2: noise augmented
if exist('data/train_noise.mat', 'file')
    subplot(2, 4, 2);
    img_noise = noise.data_train(:, :, 1, sample_idx);
    imshow(img_noise, []);
    title(sprintf('Noise Aug\nClass: %s', num2str(noise.labels_train(sample_idx))));

    subplot(2, 4, 6);
    histogram(img_noise(:), 50);
    title('Noise Aug Histogram');
    xlabel('Pixel Value');
    ylabel('Count');
    xlim([0, 1]);
end

% Sample 3: scale augmented
if exist('data/train_scale.mat', 'file')
    scale = load('data/train_scale.mat');
    subplot(2, 4, 3);
    img_scale = scale.data_train(:, :, 1, sample_idx);
    imshow(img_scale, []);
    title(sprintf('Scale Aug\nClass: %s', num2str(scale.labels_train(sample_idx))));

    subplot(2, 4, 7);
    histogram(img_scale(:), 50);
    title('Scale Aug Histogram');
    xlabel('Pixel Value');
    ylabel('Count');
    xlim([0, 1]);
end

% Sample 4: rotation augmented
if exist('data/train_rotation.mat', 'file')
    rotation = load('data/train_rotation.mat');
    subplot(2, 4, 4);
    img_rotation = rotation.data_train(:, :, 1, sample_idx);
    imshow(img_rotation, []);
    title(sprintf('Rotation Aug\nClass: %s', num2str(rotation.labels_train(sample_idx))));

    subplot(2, 4, 8);
    histogram(img_rotation(:), 50);
    title('Rotation Aug Histogram');
    xlabel('Pixel Value');
    ylabel('Count');
    xlim([0, 1]);
end

sgtitle('Data Format Comparison: Baseline vs Augmented', 'FontSize', 14, 'FontWeight', 'bold');
saveas(fig, 'output/debug_data_format_comparison.png');
fprintf('Saved visualization to: output/debug_data_format_comparison.png\n');

%% Key Finding
fprintf('\n========================================\n');
fprintf('KEY FINDING:\n');
fprintf('========================================\n');
if is_binary_baseline
    fprintf('✓ Baseline is BINARY (0/1 only)\n');
else
    fprintf('✗ Baseline is GRAYSCALE (continuous [0,1])\n');
end

if exist('data/train_noise.mat', 'file')
    if is_binary_noise
        fprintf('✓ Augmented data is BINARY (0/1 only)\n');
    else
        fprintf('✗ Augmented data is GRAYSCALE (continuous [0,1])\n');
    end

    if is_binary_baseline ~= is_binary_noise
        fprintf('\n⚠️  DATA FORMAT MISMATCH DETECTED!\n');
        fprintf('   Baseline and augmented data have DIFFERENT formats.\n');
        fprintf('   This will cause train/test distribution mismatch.\n');
    else
        fprintf('\n✓ Data formats match.\n');
    end
end

fprintf('========================================\n\n');
