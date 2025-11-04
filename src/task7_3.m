%% Task 7.3: Character Recognition on Image 1
% Apply both CNN (Task 7.1) and SOM+BoW+SVM (Task 7.2) classifiers
% to recognize characters in the original charact2.bmp image.
%
% Expected text:
%   Upper part: "HD44780A00"
%   Lower part: "ZM2"

clear; close all; clc;

%% Setup
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

addpath(genpath('src/core'));
addpath(genpath('src/utils'));

fprintf('========================================\n');
fprintf('   Task 7.3: Character Recognition\n');
fprintf('========================================\n\n');

%% Configuration
% Model paths
cnn_checkpoint = 'output/task7_1/11-04_23-37-15/cnn_best_acc.mat';
som_checkpoint = 'output/task7_2/som_model.mat';
pca_checkpoint = 'output/task7_2/pca_model.mat';
svm_checkpoint = 'output/task7_2/svm_models.mat';
task7_2_results = 'output/task7_2/results.mat';

% Image paths (NOTE: file names vs actual content)
% cropped_HD44780A00.png actually contains "HD44780A00" (lower line)
% cropped_lower.png actually contains "ZM2" (upper line)
lower_img_path = 'data/cropped_charact2/cropped_HD44780A00.png';  % HD44780A00
upper_img_path = 'data/cropped_charact2/cropped_lower.png';  % ZM2

% Output directory
output_dir = 'output/task7_3/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Class names (0-indexed in dataset)
class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};

%% Load models
fprintf('Loading models...\n');
fprintf('  CNN model: %s\n', cnn_checkpoint);
cnn_data = load(cnn_checkpoint, 'cnn');
cnn = cnn_data.cnn;

fprintf('  SOM model: %s\n', som_checkpoint);
som_data = load(som_checkpoint, 'som');
som = som_data.som;

fprintf('  PCA model: %s\n', pca_checkpoint);
pca_data = load(pca_checkpoint, 'pca_model');
pca_model = pca_data.pca_model;

fprintf('  SVM models: %s\n', svm_checkpoint);
svm_data = load(svm_checkpoint, 'models');
svm_models = svm_data.models;

fprintf('  Task 7.2 config: %s\n', task7_2_results);
task7_2_data = load(task7_2_results, 'results');
task7_2_config = task7_2_data.results.config;
fprintf('Models loaded successfully.\n\n');

%% Process Upper Part (ZM2)
fprintf('========================================\n');
fprintf('Processing Upper Part (ZM2)\n');
fprintf('========================================\n');

% Read and preprocess
img_upper = imread(upper_img_path);
if size(img_upper, 3) == 3
    img_upper = myRgb2gray(img_upper);
end

% Binarize using Otsu
threshold_upper = myOtsuThres(img_upper);
binary_upper = myImbinarize(img_upper, threshold_upper);
fprintf('Upper part binarized (Otsu threshold: %.4f)\n', threshold_upper);

% Save binary image
imwrite(binary_upper, [output_dir, 'binary_upper.png']);

% Segment characters
[chars_upper, bboxes_upper] = segment_characters(binary_upper, 800);
fprintf('Segmented %d characters from upper part\n\n', length(chars_upper));

%% Process Lower Part (HD44780A00)
fprintf('========================================\n');
fprintf('Processing Lower Part (HD44780A00)\n');
fprintf('========================================\n');

% Read and preprocess
img_lower = imread(lower_img_path);
if size(img_lower, 3) == 3
    img_lower = myRgb2gray(img_lower);
end

% Binarize using Otsu
threshold_lower = myOtsuThres(img_lower);
binary_lower = myImbinarize(img_lower, threshold_lower);
fprintf('Lower part binarized (Otsu threshold: %.4f)\n', threshold_lower);

% Save binary image
imwrite(binary_lower, [output_dir, 'binary_lower.png']);

% Segment characters
[chars_lower, bboxes_lower] = segment_characters(binary_lower, 300);
fprintf('Segmented %d characters from lower part\n\n', length(chars_lower));

%% Combine all characters
all_chars = [chars_upper, chars_lower];
num_chars = length(all_chars);
fprintf('Total characters to classify: %d\n\n', num_chars);

% Save individual character images for inspection
chars_dir = [output_dir, 'characters/'];
if ~exist(chars_dir, 'dir')
    mkdir(chars_dir);
end

for i = 1:num_chars
    char_img = all_chars{i};
    filename = sprintf('%schar_%02d.png', chars_dir, i);
    imwrite(char_img, filename);
end

%% Prepare characters for classification
fprintf('========================================\n');
fprintf('Preparing characters for classification\n');
fprintf('========================================\n');

% Resize to 64x64 for CNN and feature extraction
target_size = 64;
chars_resized = zeros(target_size, target_size, 1, num_chars);

for i = 1:num_chars
    char_img = all_chars{i};

    % Pad to square
    [h, w] = size(char_img);
    max_dim = max(h, w);
    padded = zeros(max_dim, max_dim);
    start_y = round((max_dim - h) / 2) + 1;
    start_x = round((max_dim - w) / 2) + 1;
    padded(start_y:start_y+h-1, start_x:start_x+w-1) = char_img;

    % Resize to 64x64
    resized = myImresize(padded, [target_size, target_size]);

    % Normalize to [0, 1]
    resized = double(resized);
    if max(resized(:)) > 0
        resized = resized / max(resized(:));
    end

    chars_resized(:, :, 1, i) = resized;
end

fprintf('Characters resized to %dx%d\n\n', target_size, target_size);

%% CNN Classification (Task 7.1) - WITHOUT polarity correction
fprintf('========================================\n');
fprintf('CNN Classification (BEFORE Polarity Fix)\n');
fprintf('========================================\n');

tic;
[cnn_preds_before, cnn_updated] = predict(cnn, chars_resized);
cnn_time_before = toc;

% Extract probabilities from the last layer
cnn_probs_before = cnn_updated.layers{end}.activations;

% Convert predictions to 0-indexed
cnn_preds_before = cnn_preds_before - 1;

fprintf('CNN classification completed in %.4f seconds\n', cnn_time_before);
fprintf('Average time per character: %.4f seconds\n\n', cnn_time_before / num_chars);

%% SOM+BoW+SVM Classification (Task 7.2) - WITHOUT polarity correction
fprintf('========================================\n');
fprintf('SOM+BoW+SVM (BEFORE Polarity Fix)\n');
fprintf('========================================\n');

% Extract BoW features
fprintf('Extracting BoW features...\n');
tic;
bow_features_before = extract_bow_features(chars_resized, som, task7_2_config.stride, ...
    'normalize', true, 'norm_type', 'l2', ...
    'soft_voting', task7_2_config.soft_voting, ...
    'sigma_bow', task7_2_config.sigma_bow, ...
    'spatial_pyramid', task7_2_config.spatial_pyramid, ...
    'verbose', false, ...
    'min_patch_std', task7_2_config.min_patch_std);
feature_time_before = toc;

% Apply PCA
bow_features_pca_before = (bow_features_before - pca_model.mu) * pca_model.W;

% SVM prediction
tic;
svm_preds_before = predictMulticlassSVM(svm_models, bow_features_pca_before);
svm_time_before = toc;

% Convert predictions to 0-indexed
svm_preds_before = svm_preds_before - 1;

total_svm_time_before = feature_time_before + svm_time_before;
fprintf('SVM classification completed in %.4f seconds\n\n', total_svm_time_before);

%% Apply Polarity Correction
fprintf('========================================\n');
fprintf('Applying Polarity Inversion\n');
fprintf('========================================\n');
fprintf('Inverting pixel values: x_corrected = 1 - x\n\n');

chars_corrected = 1 - chars_resized;

%% CNN Classification - AFTER polarity correction
fprintf('========================================\n');
fprintf('CNN Classification (AFTER Polarity Fix)\n');
fprintf('========================================\n');

tic;
[cnn_preds_after, cnn_updated_after] = predict(cnn, chars_corrected);
cnn_time_after = toc;

% Extract probabilities
cnn_probs_after = cnn_updated_after.layers{end}.activations;

% Convert predictions to 0-indexed
cnn_preds_after = cnn_preds_after - 1;

fprintf('CNN classification completed in %.4f seconds\n', cnn_time_after);
fprintf('Average time per character: %.4f seconds\n\n', cnn_time_after / num_chars);

%% SOM+BoW+SVM Classification - AFTER polarity correction
fprintf('========================================\n');
fprintf('SOM+BoW+SVM (AFTER Polarity Fix)\n');
fprintf('========================================\n');

% Extract BoW features
fprintf('Extracting BoW features...\n');
tic;
bow_features_after = extract_bow_features(chars_corrected, som, task7_2_config.stride, ...
    'normalize', true, 'norm_type', 'l2', ...
    'soft_voting', task7_2_config.soft_voting, ...
    'sigma_bow', task7_2_config.sigma_bow, ...
    'spatial_pyramid', task7_2_config.spatial_pyramid, ...
    'verbose', false, ...
    'min_patch_std', task7_2_config.min_patch_std);
feature_time_after = toc;

% Apply PCA
bow_features_pca_after = (bow_features_after - pca_model.mu) * pca_model.W;

% SVM prediction
tic;
svm_preds_after = predictMulticlassSVM(svm_models, bow_features_pca_after);
svm_time_after = toc;

% Convert predictions to 0-indexed
svm_preds_after = svm_preds_after - 1;

total_svm_time_after = feature_time_after + svm_time_after;
fprintf('SVM classification completed in %.4f seconds\n\n', total_svm_time_after);

%% Compute Accuracies (only on in-vocabulary characters)
% In-vocabulary positions: 1 (7 from upper), 4-13 (all from lower HD44780A00)
% Class mapping (0-indexed): 0->0, 4->1, 7->2, 8->3, A->4, D->5, H->6
in_vocab_pos = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
% Ground truth: 7, H, D, 4, 4, 7, 8, 0, A, 0, 0 -> indices: 2,6,5,1,1,2,3,0,4,0,0
ground_truth = [2, 6, 5, 1, 1, 2, 3, 0, 4, 0, 0];  % Already 0-indexed

% CNN accuracies
cnn_preds_invocab_before = cnn_preds_before(in_vocab_pos);
cnn_preds_invocab_after = cnn_preds_after(in_vocab_pos);
cnn_correct_before = sum(cnn_preds_invocab_before == ground_truth');
cnn_correct_after = sum(cnn_preds_invocab_after == ground_truth');
cnn_acc_before = cnn_correct_before / length(ground_truth) * 100;
cnn_acc_after = cnn_correct_after / length(ground_truth) * 100;

% SVM accuracies
svm_preds_invocab_before = svm_preds_before(in_vocab_pos);
svm_preds_invocab_after = svm_preds_after(in_vocab_pos);
svm_correct_before = sum(svm_preds_invocab_before == ground_truth');
svm_correct_after = sum(svm_preds_invocab_after == ground_truth');
svm_acc_before = svm_correct_before / length(ground_truth) * 100;
svm_acc_after = svm_correct_after / length(ground_truth) * 100;

%% Display Results
fprintf('========================================\n');
fprintf('Results Summary\n');
fprintf('========================================\n\n');

fprintf('BEFORE Polarity Fix (Inverted Input):\n');
fprintf('  CNN:           %d/11 correct (%.1f%%)\n', cnn_correct_before, cnn_acc_before);
fprintf('  SOM+BoW+SVM:   %d/11 correct (%.1f%%)\n\n', svm_correct_before, svm_acc_before);

fprintf('AFTER Polarity Fix (Corrected Input):\n');
fprintf('  CNN:           %d/11 correct (%.1f%%)\n', cnn_correct_after, cnn_acc_after);
fprintf('  SOM+BoW+SVM:   %d/11 correct (%.1f%%)\n\n', svm_correct_after, svm_acc_after);

fprintf('Improvement from Polarity Fix:\n');
fprintf('  CNN:           %+.1f%% (%.1f%% → %.1f%%)\n', cnn_acc_after - cnn_acc_before, cnn_acc_before, cnn_acc_after);
fprintf('  SOM+BoW+SVM:   %+.1f%% (%.1f%% → %.1f%%)\n\n', svm_acc_after - svm_acc_before, svm_acc_before, svm_acc_after);

fprintf('Expected Text (in-vocabulary only): 7 HD44780A00\n');
fprintf('  CNN (before):  %s\n', strjoin(class_names(cnn_preds_before(in_vocab_pos) + 1), ' '));
fprintf('  CNN (after):   %s\n', strjoin(class_names(cnn_preds_after(in_vocab_pos) + 1), ' '));
fprintf('  SVM (before):  %s\n', strjoin(class_names(svm_preds_before(in_vocab_pos) + 1), ' '));
fprintf('  SVM (after):   %s\n', strjoin(class_names(svm_preds_after(in_vocab_pos) + 1), ' '));

%% Visualization: Polarity comparison
fprintf('\n========================================\n');
fprintf('Generating Visualizations\n');
fprintf('========================================\n');

% Load training set samples for comparison
train_data = load('data/train.mat');
train_images = train_data.data_train;
train_labels = train_data.labels_train;

% Select 3 samples from training set (classes 0, 7, H for diversity)
sample_classes = [0, 2, 6];  % 0-indexed: 0, 7, H
train_samples = cell(1, 3);
for i = 1:3
    cls = sample_classes(i);
    idx = find(train_labels == cls, 1);
    train_samples{i} = train_images(:, :, 1, idx);
end

% Create polarity comparison figure with minimal whitespace
fig = figure('Position', [100, 100, 1200, 280], 'Color', 'white', 'Visible', 'off');

% Left: Test image (inverted polarity)
subplot('Position', [0.04, 0.12, 0.45, 0.78]);  % [left, bottom, width, height]
imshow(binary_lower);
title('Test Image (Inverted Polarity)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'black');
text(size(binary_lower, 2)/2, size(binary_lower, 1)+12, 'Black background, white foreground', ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'black');

% Right: Training samples (normal polarity)
subplot('Position', [0.52, 0.12, 0.45, 0.78]);  % Minimal gap
montage_img = zeros(64, 64*3);
for i = 1:3
    montage_img(:, (i-1)*64+1:i*64) = train_samples{i};
end
imshow(montage_img);
title('Training Samples (Normal Polarity)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'black');
text(size(montage_img, 2)/2, size(montage_img, 1)+12, 'White background, black foreground', ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'black');

saveas(fig, [output_dir, 'polarity_comparison.png']);
close(fig);

fprintf('Visualizations saved.\n\n');

%% Save Results
results = struct();
results.cnn_predictions_before = cnn_preds_before;
results.cnn_predictions_after = cnn_preds_after;
results.cnn_acc_before = cnn_acc_before;
results.cnn_acc_after = cnn_acc_after;
results.svm_predictions_before = svm_preds_before;
results.svm_predictions_after = svm_preds_after;
results.svm_acc_before = svm_acc_before;
results.svm_acc_after = svm_acc_after;
results.in_vocab_positions = in_vocab_pos;
results.ground_truth = ground_truth;
results.class_names = class_names;

save([output_dir, 'results.mat'], 'results');

fprintf('========================================\n');
fprintf('Task 7.3 Complete\n');
fprintf('========================================\n');
fprintf('Results saved to: %s\n', output_dir);

%% Helper function: segment_characters
function [chars, bboxes] = segment_characters(binary_img, min_area)
    % Find connected components
    CC = myBwconncomp(binary_img);

    % Extract region properties
    props = myRegionprops(CC, 'BoundingBox', 'Area');

    % Filter by area
    valid_idx = [];
    for i = 1:length(props)
        if props(i).Area >= min_area
            valid_idx = [valid_idx, i]; %#ok<AGROW>
        end
    end

    % Sort by x-coordinate (left to right)
    bboxes_all = zeros(length(valid_idx), 4);
    for i = 1:length(valid_idx)
        bboxes_all(i, :) = props(valid_idx(i)).BoundingBox;
    end

    [~, sort_idx] = sort(bboxes_all(:, 1));
    sorted_bboxes = bboxes_all(sort_idx, :);
    sorted_idx = valid_idx(sort_idx);

    % Segment characters (split merged characters if needed)
    chars = {};
    bboxes = [];

    min_width = min(sorted_bboxes(:, 3));
    merge_threshold = min_width * 1.8;

    for i = 1:size(sorted_bboxes, 1)
        bbox = sorted_bboxes(i, :);
        x = round(bbox(1));
        y = round(bbox(2));
        w = round(bbox(3));
        h = round(bbox(4));

        char_region = binary_img(y:y+h-1, x:x+w-1);

        % Check if merged character
        if w > merge_threshold
            split_point = round(w / 2);

            char1 = char_region(:, 1:split_point);
            char2 = char_region(:, split_point+1:end);

            chars{end+1} = char1; %#ok<AGROW>
            bboxes = [bboxes; x, y, split_point, h]; %#ok<AGROW>

            chars{end+1} = char2; %#ok<AGROW>
            bboxes = [bboxes; x+split_point, y, w-split_point, h]; %#ok<AGROW>
        else
            chars{end+1} = char_region; %#ok<AGROW>
            bboxes = [bboxes; bbox]; %#ok<AGROW>
        end
    end
end
