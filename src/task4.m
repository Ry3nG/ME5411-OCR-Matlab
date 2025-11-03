clear; close all; clc;

% Get the project root directory
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

addpath(genpath('src/core'));
addpath(genpath('src/utils'));

% Input and output directories
croppedImgPath = 'output/task3/cropped_HD44780A00.png';
outputDir = 'output/task4/';

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Read the cropped image from Task 3
img = imread(croppedImgPath);
if size(img, 3) == 3
    img = rgb2gray(img);
end

% Save grayscale input image
imwrite(img, [outputDir, 'grayscale_input.png']);

% Calculate Otsu threshold
threshold = myOtsuThres(img);
fprintf('Otsu threshold calculated: %.4f\n', threshold);

% Apply binarization using Otsu threshold
binaryImg = myImbinarize(img, threshold);

% Save binary image
imwrite(binaryImg, [outputDir, 'binary_otsu.png']);

fprintf('\nTask 4 completed successfully!\n');
fprintf('Binary image saved to: %s\n', outputDir);
fprintf('Otsu threshold: %.4f (normalized) or %.1f (0-255 scale)\n', threshold, threshold * 255);
