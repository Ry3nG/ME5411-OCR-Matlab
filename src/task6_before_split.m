% Temporary script to generate "before splitting" visualization for report
clear; close all; clc;

addpath(genpath('core'));
addpath(genpath('utils'));

% Input
binaryImgPath = '../output/task4/binary_otsu.png';
outputDir = '../output/task6/';

% Read the binary image
binaryImg = imread(binaryImgPath);
if size(binaryImg, 3) == 3
    binaryImg = rgb2gray(binaryImg);
end
if ~islogical(binaryImg)
    binaryImg = binaryImg > 128;
end

% Connected Component Analysis
CC = myBwconncomp(binaryImg);
props = myRegionprops(CC, 'BoundingBox', 'Area');

% Filter noise
min_area = 800;
valid_idx = [];
for i = 1:length(props)
    if props(i).Area >= min_area
        valid_idx = [valid_idx, i];
    end
end

% Sort by x-coordinate
bboxes = zeros(length(valid_idx), 4);
for i = 1:length(valid_idx)
    bboxes(i, :) = props(valid_idx(i)).BoundingBox;
end
[~, sort_idx] = sort(bboxes(:, 1));
sorted_bboxes = bboxes(sort_idx, :);

% Visualize WITHOUT splitting (to show the problem)
figure('Name', 'Before Width-Based Splitting', 'Position', [100, 100, 900, 300]);
imshow(binaryImg);
hold on;

% Draw bounding boxes
for i = 1:size(sorted_bboxes, 1)
    bbox = sorted_bboxes(i, :);
    rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
    text(bbox(1), bbox(2)-5, sprintf('%d', i), ...
        'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');
end
hold off;
title(sprintf('Before Width-Based Splitting (%d components - merged characters visible)', size(sorted_bboxes, 1)));

% Save visualization
set(gca, 'Position', [0 0 1 1]);
set(gcf, 'PaperPositionMode', 'auto');
print([outputDir, 'before_splitting.png'], '-dpng', '-r150');

fprintf('Generated before-splitting visualization with %d components\n', size(sorted_bboxes, 1));
fprintf('Merged pairs: Component 2 (44), Component 5 (80), Component 7 (00)\n');
