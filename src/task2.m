clear; close all; clc;

% Get the project root directory
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

addpath(genpath('src/core'));
addpath(genpath('src/utils'));

imgPath = 'canvas/charact2.bmp';
outputDir = 'output/task2/';

% Read and convert image to grayscale if needed
img = imread(imgPath);
if size(img, 3) == 3
    img = rgb2gray(img);
end

% Save original image
imwrite(img, [outputDir, 'original.png']);

% Define filter sizes to experiment with
% 5x5 (required), 15x15 (medium), 31x31 (large)
filterSizes = [5, 15, 31];

% Apply averaging filters of different sizes and save individual results
for i = 1:length(filterSizes)
    filterSize = filterSizes(i);
    fprintf('Applying %dx%d averaging filter...\n', filterSize, filterSize);

    % Apply custom averaging filter
    filteredImg = myAverageFilter(img, filterSize);

    % Save individual result
    imwrite(filteredImg, [outputDir, sprintf('average_filter_%dx%d.png', filterSize, filterSize)]);
end

% No need for combined comparison figures - individual images will be used in report

fprintf('\nTask 2 completed successfully!\n');
fprintf('Results saved to: %s\n', outputDir);
fprintf('\nFilter sizes tested:\n');
for i = 1:length(filterSizes)
    fprintf('  - %dx%d averaging filter\n', filterSizes(i), filterSizes(i));
end
