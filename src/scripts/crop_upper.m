clear; close all; clc;

% Get the project root directory
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

addpath(genpath('src/core'));
addpath(genpath('src/utils'));

imgPath = 'canvas/charact2.bmp';
outputDirRaw = '/home/gong-zerui/code/ME5411/ME5411-OCR-Matlab/output/task7_3/raw/';

% Create output directory if it doesn't exist
if ~exist(outputDirRaw, 'dir')
    mkdir(outputDirRaw);
end

% Read and convert image to grayscale if needed
img = imread(imgPath);
if size(img, 3) == 3
    img = myRgb2gray(img);
end

% Calculate average character width from task6
charDir = 'output/task6/characters/';
charFiles = dir([charDir, 'char_*.png']);
numChars = length(charFiles);
widths = zeros(numChars, 1);
for i = 1:numChars
    charImg = imread([charDir, charFiles(i).name]);
    [~, width] = size(charImg);
    widths(i) = width;
end
avgWidth = mean(widths);
targetWidth = round(avgWidth * 3);  % Width for 3 characters

% Define cropping rectangle for upper portion (7M2 region)
% rect = [x, y, width, height]
% Image size is 990 x 367, upper portion is in the top portion
% x=290 to reduce left gap, width reduced to remove right gap
rectUpper = [290, 50, targetWidth + 15, 150];

% Crop the upper portion using custom function
croppedImgUpper = myImcrop(img, rectUpper);

% Save cropped upper portion
imwrite(croppedImgUpper, [outputDirRaw, 'cropped_upper.png']);

fprintf('\nUpper portion cropped successfully!\n');
fprintf('Cropped image saved to: %s\n', outputDirRaw);
fprintf('Crop parameters: [x=%d, y=%d, width=%d, height=%d]\n', rectUpper(1), rectUpper(2), rectUpper(3), rectUpper(4));

