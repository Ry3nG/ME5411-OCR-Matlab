clear; close all; clc;

addpath(genpath('core'));
addpath(genpath('utils'));

% Input and output directories
binaryImgPath = '../output/task4/binary_otsu.png';
outputDir = '../output/task6/';
charactersDir = [outputDir, 'characters/'];

% Create output directories
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
if ~exist(charactersDir, 'dir')
    mkdir(charactersDir);
end

% Read the binary image from Task 4
binaryImg = imread(binaryImgPath);
if size(binaryImg, 3) == 3
    binaryImg = rgb2gray(binaryImg);
end

% Convert to logical
if ~islogical(binaryImg)
    binaryImg = binaryImg > 128;
end

fprintf('Starting character segmentation...\n');

%% Step 1: Connected Component Analysis
fprintf('Finding connected components...\n');
CC = myBwconncomp(binaryImg);
fprintf('Found %d connected components\n', CC.NumObjects);

%% Step 2: Extract region properties
props = myRegionprops(CC, 'BoundingBox', 'Area');

%% Step 3: Filter noise (small components)
min_area = 800;  % Minimum area threshold (increased to filter more noise)
valid_idx = [];
for i = 1:length(props)
    if props(i).Area >= min_area
        valid_idx = [valid_idx, i];
    end
end

fprintf('After filtering: %d valid components\n', length(valid_idx));

%% Step 4: Sort components by x-coordinate (left to right)
bboxes = zeros(length(valid_idx), 4);
for i = 1:length(valid_idx)
    bboxes(i, :) = props(valid_idx(i)).BoundingBox;
end

% Sort by x-coordinate (first column)
[~, sort_idx] = sort(bboxes(:, 1));
sorted_bboxes = bboxes(sort_idx, :);
sorted_idx = valid_idx(sort_idx);

%% Step 5: Segment characters using width-based splitting for merged characters
segmented_chars = {};
char_bboxes = [];

% Calculate minimum width from all valid components
min_width = min(sorted_bboxes(:, 3));
fprintf('Minimum character width: %.1f\n', min_width);

% Threshold for detecting merged characters (1.8 times minimum width)
merge_threshold = min_width * 1.8;

for i = 1:size(sorted_bboxes, 1)
    bbox = sorted_bboxes(i, :);
    x = round(bbox(1));
    y = round(bbox(2));
    w = round(bbox(3));
    h = round(bbox(4));

    % Extract character region
    char_region = binaryImg(y:y+h-1, x:x+w-1);

    % Check if this is a merged character (width > 1.8 * min_width)
    if w > merge_threshold
        % Split horizontally into two equal parts
        split_point = round(w / 2);

        fprintf('Splitting merged character %d (width=%d > threshold=%.1f) at midpoint %d\n', ...
            i, w, merge_threshold, split_point);

        % Split into two characters
        char1 = char_region(:, 1:split_point);
        char2 = char_region(:, split_point+1:end);

        segmented_chars{end+1} = char1;
        char_bboxes = [char_bboxes; x, y, split_point, h];

        segmented_chars{end+1} = char2;
        char_bboxes = [char_bboxes; x+split_point, y, w-split_point, h];
    else
        % Normal width character, keep as is
        segmented_chars{end+1} = char_region;
        char_bboxes = [char_bboxes; bbox];
    end
end

fprintf('Total segmented characters: %d\n', length(segmented_chars));

%% Step 6: Save individual characters
for i = 1:length(segmented_chars)
    char_img = segmented_chars{i};
    filename = sprintf('%schar_%02d.png', charactersDir, i);
    imwrite(char_img, filename);
end

%% Step 7: Visualize segmentation with bounding boxes
figure('Name', 'Character Segmentation', 'Position', [100, 100, 900, 300]);
imshow(binaryImg);
hold on;

% Draw bounding boxes
for i = 1:size(char_bboxes, 1)
    bbox = char_bboxes(i, :);
    rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);

    % Add character number
    text(bbox(1), bbox(2)-5, sprintf('%d', i), ...
        'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');
end
hold off;
title(sprintf('Character Segmentation (%d characters)', length(segmented_chars)));

% Save visualization
set(gca, 'Position', [0 0 1 1]);
set(gcf, 'PaperPositionMode', 'auto');
print([outputDir, 'segmentation_result.png'], '-dpng', '-r150');

%% Step 8: Create montage of all characters
figure('Name', 'Segmented Characters', 'Position', [100, 100, 1200, 200]);

num_chars = length(segmented_chars);
for i = 1:num_chars
    subplot(1, num_chars, i);
    imshow(segmented_chars{i});
    title(sprintf('Char %d', i));
end

saveas(gcf, [outputDir, 'characters_montage.png']);

fprintf('\nTask 6 completed successfully!\n');
fprintf('Segmented %d characters\n', length(segmented_chars));
fprintf('Results saved to: %s\n', outputDir);
