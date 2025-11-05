% task8_generate_augmented_datasets.m
% Generate augmented datasets for Task 8 sensitivity analysis
%
% Generates 4 augmented versions of training data:
%   - train_noise.mat: Salt-and-pepper + Gaussian noise
%   - train_scale.mat: Random scaling (0.9-1.1x)
%   - train_rotation.mat: Random rotation (±10°)
%   - train_combined.mat: All augmentations combined
%
% All images are 64x64 to match existing train.mat format
%
% Usage:
%   Run from project root: matlab -batch "run('src/task8_generate_augmented_datasets.m')"

clear all; %#ok<CLALL>
close all;

fprintf('\n');
fprintf('=========================================\n');
fprintf('  Task 8: Generate Augmented Datasets   \n');
fprintf('=========================================\n\n');

%% Setup
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

% Add paths
addpath(genpath('src/core'));
addpath(genpath('src/utils'));

% Set random seed for reproducibility
rng(42);

%% Configuration
IMG_DIM = 64;  % Final image dimension (64x64)
VISUALIZE = true;
NUM_VIS_SAMPLES = 7;  % Visualize 7 samples (one per class)

% Output paths
data_dir = fullfile(project_root, 'data');
output_dir = fullfile(project_root, 'output', 'task8_augmentation');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('Configuration:\n');
fprintf('  Image dimension: %dx%d\n', IMG_DIM, IMG_DIM);
fprintf('  Random seed: 42 (for reproducibility)\n');
fprintf('  Output directory: %s\n', data_dir);
fprintf('  Visualization: %s\n\n', mat2str(VISUALIZE));

%% Load original dataset
fprintf('Loading original dataset from dataset_2025...\n');
dataset_path = fullfile(data_dir, 'dataset_2025');

labels_name = {'0', '4', '7', '8', 'A', 'D', 'H'};
train_ratio = 0.75;

% Count total images
total_images = 0;
for i = 1:length(labels_name)
    class_dir = fullfile(dataset_path, labels_name{i});
    files = dir(fullfile(class_dir, '*.png'));
    num_train = floor(length(files) * train_ratio);
    total_images = total_images + num_train;
end

fprintf('  Total training images: %d\n\n', total_images);

%% Define augmentation strategies
strategies = {
    'noise',     'Gaussian noise (σ=0.05) + Salt-and-pepper (2%)'
    'scale',     'Random scaling (0.8-1.2x)'
    'rotation',  'Random rotation (±20°)'
    'combined',  'Noise + Scale + Rotation + Translation'
};

fprintf('Augmentation strategies:\n');
for i = 1:size(strategies, 1)
    fprintf('  %d. %-10s: %s\n', i, strategies{i, 1}, strategies{i, 2});
end
fprintf('\n');

%% Process each augmentation strategy
for strategy_idx = 1:size(strategies, 1)
    strategy_name = strategies{strategy_idx, 1};
    strategy_desc = strategies{strategy_idx, 2};

    % Check if already exists
    output_file = fullfile(data_dir, sprintf('train_%s.mat', strategy_name));
    if exist(output_file, 'file')
        fprintf('=========================================\n');
        fprintf('Skipping: %s (already exists)\n', strategy_name);
        fprintf('File: %s\n', output_file);
        fprintf('=========================================\n\n');
        continue;
    end

    fprintf('=========================================\n');
    fprintf('Processing: %s\n', strategy_name);
    fprintf('Description: %s\n', strategy_desc);
    fprintf('=========================================\n');

    % Pre-allocate
    data_train = zeros(IMG_DIM, IMG_DIM, 1, total_images, 'single');
    labels_train = zeros(total_images, 1);

    train_idx = 1;
    vis_samples = [];
    vis_labels = [];
    vis_collected = false(length(labels_name), 1);

    % Process each class
    for class_idx = 1:length(labels_name)
        class_name = labels_name{class_idx};
        class_dir = fullfile(dataset_path, class_name);
        files = dir(fullfile(class_dir, '*.png'));

        num_train = floor(length(files) * train_ratio);

        fprintf('  Processing class %s: %d training images\n', class_name, num_train);

        for file_idx = 1:num_train
            % Read image
            img = imread(fullfile(class_dir, files(file_idx).name));

            % Convert to grayscale if needed
            if size(img, 3) == 3
                img = myRgb2gray(img);
            end

            % Apply augmentation based on strategy
            img_aug = apply_augmentation(img, strategy_name);

            % Resize to 64x64 (matches Task 7.1 loadDataset.m line 74)
            img_aug = myImresize(img_aug, [IMG_DIM, IMG_DIM], 'bilinear');

            % Normalize to [0, 1] WITHOUT binarization (matches Task 7.1 loadDataset.m line 75)
            % This preserves grayscale information and matches train.mat format
            img_aug = double(img_aug) / 255.0;

            % Store
            data_train(:, :, 1, train_idx) = img_aug;
            labels_train(train_idx) = labelChar2Num(class_name);
            train_idx = train_idx + 1;

            % Collect one sample per class for visualization
            if VISUALIZE && ~vis_collected(class_idx)
                vis_samples = cat(4, vis_samples, img_aug);
                vis_labels = [vis_labels; labelChar2Num(class_name)]; %#ok<AGROW>
                vis_collected(class_idx) = true;
            end

            % Progress indicator
            if mod(train_idx, 500) == 0
                fprintf('    Progress: %d/%d (%.1f%%)\n', train_idx, total_images, ...
                    100*train_idx/total_images);
            end
        end
    end

    fprintf('  Completed: %d images processed\n', train_idx - 1);

    % Save augmented dataset
    output_file = fullfile(data_dir, sprintf('train_%s.mat', strategy_name));
    fprintf('  Saving to: %s\n', output_file);
    save(output_file, 'data_train', 'labels_train', '-v7.3');

    file_info = dir(output_file);
    fprintf('  File size: %.2f MB\n', file_info.bytes / 1024 / 1024);

    % Visualize samples
    if VISUALIZE
        visualize_augmentation(vis_samples, vis_labels, strategy_name, ...
            strategy_desc, output_dir);
    end

    fprintf('\n');
end

%% Create comparison visualization
fprintf('=========================================\n');
fprintf('Creating comparison visualization...\n');
fprintf('=========================================\n');

% Load all datasets including baseline
baseline = load(fullfile(data_dir, 'train.mat'));
augmented_data = cell(1, 5);
augmented_data{1} = baseline.data_train;

for i = 1:size(strategies, 1)
    strategy_name = strategies{i, 1};
    aug_file = fullfile(data_dir, sprintf('train_%s.mat', strategy_name));
    loaded = load(aug_file);
    augmented_data{i+1} = loaded.data_train;
end

% Create comparison figure for each class
create_comparison_figure(augmented_data, baseline.labels_train, ...
    labels_name, strategies, output_dir);

fprintf('\n');
fprintf('=========================================\n');
fprintf('✓ All augmented datasets generated!\n');
fprintf('=========================================\n');
fprintf('\nGenerated files:\n');
fprintf('  data/train.mat (baseline - already exists)\n');
for i = 1:size(strategies, 1)
    fprintf('  data/train_%s.mat\n', strategies{i, 1});
end
fprintf('\nVisualization saved to: %s\n', output_dir);
fprintf('\nPlease review the visualization before proceeding with experiments.\n\n');

%% Helper Functions

function img_aug = apply_augmentation(img, strategy)
    % Apply augmentation based on strategy
    % Input: img (uint8 or double), strategy (string)
    % Output: img_aug (double, [0-255] range)

    img = double(img);

    switch strategy
        case 'noise'
            img_aug = augment_noise(img);

        case 'scale'
            img_aug = augment_scale(img);

        case 'rotation'
            img_aug = augment_rotation(img);

        case 'combined'
            % Apply all augmentations in sequence
            img_aug = augment_scale(img);
            img_aug = augment_rotation(img_aug);
            img_aug = augment_translation(img_aug);
            img_aug = augment_noise(img_aug);

        otherwise
            error('Unknown augmentation strategy: %s', strategy);
    end

    % Ensure output is in [0, 255] range
    img_aug = max(0, min(255, img_aug));
end

function img_aug = augment_noise(img)
    % Add noise to grayscale image (preserving grayscale values)
    % Uses Gaussian noise + salt-and-pepper to simulate real-world degradation

    % Normalize input to [0, 1] range
    if max(img(:)) > 1
        img = double(img) / 255.0;
    else
        img = double(img);
    end

    [h, w] = size(img);

    % Add Gaussian noise (σ=0.05, moderate noise)
    % This preserves grayscale information while adding realistic noise
    gaussian_noise = randn(h, w) * 0.05;
    img_noisy = img + gaussian_noise;

    % Add sparse salt-and-pepper noise (2% density)
    % Salt (white spots)
    salt_mask = rand(h, w) < 0.01;
    img_noisy(salt_mask) = 1.0;

    % Pepper (black spots)
    pepper_mask = rand(h, w) < 0.01;
    img_noisy(pepper_mask) = 0.0;

    % Clip to [0, 1] range
    img_noisy = max(0, min(1, img_noisy));

    % Return in [0, 255] range (will be normalized to [0,1] later)
    img_aug = img_noisy * 255.0;
end

function img_aug = augment_scale(img)
    % Random scaling (0.8-1.2x, STRONGER)

    scale = 0.8 + 0.4 * rand();  % Uniform(0.8, 1.2)

    [h, w] = size(img);
    new_h = round(h * scale);
    new_w = round(w * scale);

    % Resize
    img_scaled = myImresize(img, [new_h, new_w], 'bilinear');

    % Crop or pad to original size
    img_aug = crop_or_pad(img_scaled, [h, w]);
end

function img_aug = augment_rotation(img)
    % Random rotation (±20°, STRONGER)

    angle = -20 + 40 * rand();  % Uniform(-20, 20)

    % Rotate (using loose mode to capture full rotated image)
    img_rotated = myImrotate(img, angle, 'bilinear', 'loose');

    % Crop or pad to original size
    [h, w] = size(img);
    img_aug = crop_or_pad(img_rotated, [h, w]);
end

function img_aug = augment_translation(img)
    % Random translation (±5%)

    [h, w] = size(img);
    tx = round((-0.05 + 0.1 * rand()) * w);  % ±5% of width
    ty = round((-0.05 + 0.1 * rand()) * h);  % ±5% of height

    % Create translated image (fill with white background = 255)
    img_aug = ones(h, w) * 255;

    % Calculate source and destination indices
    src_x1 = max(1, 1 - tx);
    src_x2 = min(w, w - tx);
    src_y1 = max(1, 1 - ty);
    src_y2 = min(h, h - ty);

    dst_x1 = max(1, 1 + tx);
    dst_x2 = min(w, w + tx);
    dst_y1 = max(1, 1 + ty);
    dst_y2 = min(h, h + ty);

    % Copy translated region
    img_aug(dst_y1:dst_y2, dst_x1:dst_x2) = img(src_y1:src_y2, src_x1:src_x2);
end

function img_out = crop_or_pad(img, target_size)
    % Crop center or pad to target size
    % Fill padding with white (255)

    [h, w] = size(img);
    [target_h, target_w] = deal(target_size(1), target_size(2));

    img_out = ones(target_h, target_w) * 255;

    % Calculate crop/pad region
    if h >= target_h && w >= target_w
        % Crop center
        y1 = floor((h - target_h) / 2) + 1;
        x1 = floor((w - target_w) / 2) + 1;
        img_out = img(y1:y1+target_h-1, x1:x1+target_w-1);
    elseif h <= target_h && w <= target_w
        % Pad center
        y1 = floor((target_h - h) / 2) + 1;
        x1 = floor((target_w - w) / 2) + 1;
        img_out(y1:y1+h-1, x1:x1+w-1) = img;
    else
        % Mixed: crop one dimension, pad the other
        % Crop/pad height
        if h > target_h
            y_src = floor((h - target_h) / 2) + 1;
            y_dst = 1;
            h_copy = target_h;
        else
            y_src = 1;
            y_dst = floor((target_h - h) / 2) + 1;
            h_copy = h;
        end

        % Crop/pad width
        if w > target_w
            x_src = floor((w - target_w) / 2) + 1;
            x_dst = 1;
            w_copy = target_w;
        else
            x_src = 1;
            x_dst = floor((target_w - w) / 2) + 1;
            w_copy = w;
        end

        img_out(y_dst:y_dst+h_copy-1, x_dst:x_dst+w_copy-1) = ...
            img(y_src:y_src+h_copy-1, x_src:x_src+w_copy-1);
    end
end

function visualize_augmentation(samples, labels, strategy_name, strategy_desc, output_dir)
    % Visualize augmentation samples (one per class)
    % WHITE BACKGROUND as per CLAUDE.md requirements

    labels_name = {'0', '4', '7', '8', 'A', 'D', 'H'};

    % Create subfolder for individual images
    subfolder = fullfile(output_dir, strategy_name);
    if ~exist(subfolder, 'dir')
        mkdir(subfolder);
    end

    % Create combined figure with WHITE background
    fig = figure('Position', [100, 100, 1400, 300], 'Color', 'w');

    for i = 1:size(samples, 4)
        subplot(1, 7, i);
        imshow(samples(:, :, 1, i), []);
        title(sprintf('Class: %s', labels_name{labels(i) + 1}), 'FontSize', 10);

        % Save individual image (white background, black character)
        individual_file = fullfile(subfolder, sprintf('class_%s.png', labels_name{labels(i) + 1}));
        imwrite(samples(:, :, 1, i), individual_file);
    end

    sgtitle(sprintf('Augmentation: %s - %s', strategy_name, strategy_desc), ...
        'FontSize', 12, 'FontWeight', 'bold');

    % Save combined figure with white background
    output_file = fullfile(output_dir, sprintf('aug_%s_samples.png', strategy_name));
    saveas(gcf, output_file);
    fprintf('  Saved visualization: %s\n', output_file);
    fprintf('  Saved individual images to: %s/\n', subfolder);
    close(gcf);
end

function create_comparison_figure(augmented_data, labels, labels_name, strategies, output_dir)
    % Create comparison figure showing all augmentation types
    % Pick one sample from each class

    num_classes = length(labels_name);
    num_strategies = length(augmented_data);

    figure('Position', [50, 50, 1600, 900]);

    % For each class, find one sample and show all augmentations
    for class_idx = 1:num_classes
        % Find first sample of this class
        sample_idx = find(labels == class_idx, 1);

        for strategy_idx = 1:num_strategies
            subplot(num_classes, num_strategies, (class_idx-1)*num_strategies + strategy_idx);

            img = augmented_data{strategy_idx}(:, :, 1, sample_idx);
            img = squeeze(img);  % Remove singleton dims

            % Ensure it's 2D
            if ndims(img) > 2
                img = img(:, :, 1);
            end

            imshow(img, []);
            axis image off;

            % Add title for first row
            if class_idx == 1
                if strategy_idx == 1
                    title('Baseline', 'FontSize', 10, 'FontWeight', 'bold');
                else
                    title(strategies{strategy_idx-1, 1}, 'FontSize', 10, 'FontWeight', 'bold');
                end
            end

            % Add ylabel for first column
            if strategy_idx == 1
                ylabel(sprintf('Class: %s', labels_name{class_idx}), ...
                    'FontSize', 10, 'FontWeight', 'bold');
            end
        end
    end

    sgtitle('Augmentation Strategy Comparison (All Classes)', ...
        'FontSize', 14, 'FontWeight', 'bold');

    % Save figure
    output_file = fullfile(output_dir, 'augmentation_comparison_all.png');
    saveas(gcf, output_file);
    fprintf('Saved comparison figure: %s\n', output_file);
    close(gcf);
end
