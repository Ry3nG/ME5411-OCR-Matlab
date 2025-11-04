clear; close all; clc;

% Get the project root directory
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

addpath(genpath('src/core'));
addpath(genpath('src/utils'));

imgPath = 'canvas/charact2.bmp';
outputDir = 'output/task1/';

img = imread(imgPath);
if size(img, 3) == 3
    img = myRgb2gray(img);
end

% Save original image
imwrite(img, [outputDir, 'original.png']);

% Histogram equalization
imgHistEq = myHistEq(img);
imwrite(imgHistEq, [outputDir, 'histogram_equalization.png']);

% Contrast stretching
imgStretch = myContrastStretch(img);
imwrite(imgStretch, [outputDir, 'contrast_stretching.png']);

% Gamma correction for different gamma values
gamma_values = [0.5, 1.5, 2.0];
for i = 1:length(gamma_values)
    gamma = gamma_values(i);
    imgGamma = myImadjust(img, [], [], [], [], gamma);
    imwrite(imgGamma, [outputDir, sprintf('gamma_%.1f.png', gamma)]);
end

fprintf('Task 1 completed. Results saved to %s\n', outputDir);
