clear; close all; clc;

addpath(genpath('core'));
addpath(genpath('utils'));

% Input and output directories
croppedImgPath = '../output/task3/cropped_HD44780A00.png';
outputDir = '../output/task4/';

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Read the cropped image from Task 3
img = imread(croppedImgPath);
if size(img, 3) == 3
    img = rgb2gray(img);
end

% Display original cropped image
figure('Name', 'Task 4: Grayscale Input', 'Position', [100, 100, 900, 300]);
imshow(img);
title('Cropped Image (Grayscale)');
imwrite(img, [outputDir, 'grayscale_input.png']);

% Calculate Otsu threshold
threshold = myOtsuThres(img);
fprintf('Otsu threshold calculated: %.4f\n', threshold);

% Apply binarization using Otsu threshold
binaryImg = myImbinarize(img, threshold);

% Display binary image
figure('Name', 'Task 4: Binary Image (Otsu)', 'Position', [100, 100, 900, 300]);
imshow(binaryImg);
title(sprintf('Binary Image (Otsu threshold = %.4f)', threshold));
imwrite(binaryImg, [outputDir, 'binary_otsu.png']);

% Create comparison figure
figure('Name', 'Task 4: Comparison', 'Position', [100, 100, 1600, 400]);

subplot(1, 2, 1);
imshow(img);
title('Grayscale Image');

subplot(1, 2, 2);
imshow(binaryImg);
title(sprintf('Binary Image (Otsu \\theta=%.3f)', threshold));

saveas(gcf, [outputDir, 'comparison.png']);

fprintf('\nTask 4 completed successfully!\n');
fprintf('Binary image saved to: %s\n', outputDir);
fprintf('Otsu threshold: %.4f (normalized) or %.1f (0-255 scale)\n', threshold, threshold * 255);
