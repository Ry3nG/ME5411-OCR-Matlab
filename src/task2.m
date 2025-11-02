clear; close all; clc;

addpath(genpath('core'));
addpath(genpath('utils'));

imgPath = '../canvas/charact2.bmp';
outputDir = '../output/task2/';

% Read and convert image to grayscale if needed
img = imread(imgPath);
if size(img, 3) == 3
    img = rgb2gray(img);
end

% Display and save original image
figure('Name', 'Task 2: Original Image', 'Position', [100, 100, 800, 600]);
imshow(img);
title('Original Image');
imwrite(img, [outputDir, 'original.png']);

% Define filter sizes to experiment with
% 5x5 (required), 15x15 (medium), 31x31 (large)
filterSizes = [5, 15, 31];

% Apply averaging filters of different sizes
filteredImages = cell(1, length(filterSizes));

for i = 1:length(filterSizes)
    filterSize = filterSizes(i);
    fprintf('Applying %dx%d averaging filter...\n', filterSize, filterSize);

    % Apply custom averaging filter
    filteredImg = myAverageFilter(img, filterSize);
    filteredImages{i} = filteredImg;

    % Display and save individual result
    figure('Name', sprintf('Averaging Filter %dx%d', filterSize, filterSize), ...
           'Position', [100, 100, 800, 600]);
    imshow(filteredImg);
    title(sprintf('Averaging Filter %d\\times%d', filterSize, filterSize));
    imwrite(filteredImg, [outputDir, sprintf('average_filter_%dx%d.png', filterSize, filterSize)]);
end

% No need for combined comparison figures - individual images will be used in report

fprintf('\nTask 2 completed successfully!\n');
fprintf('Results saved to: %s\n', outputDir);
fprintf('\nFilter sizes tested:\n');
for i = 1:length(filterSizes)
    fprintf('  - %dx%d averaging filter\n', filterSizes(i), filterSizes(i));
end
