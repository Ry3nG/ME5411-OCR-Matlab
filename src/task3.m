clear; close all; clc;

addpath(genpath('core'));
addpath(genpath('utils'));

imgPath = '../canvas/charact2.bmp';
outputDir = '../output/task3/';

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Read and convert image to grayscale if needed
img = imread(imgPath);
if size(img, 3) == 3
    img = rgb2gray(img);
end

% Save original image
imwrite(img, [outputDir, 'original.png']);

% Define cropping rectangle for HD44780A00 line
% rect = [x, y, width, height]
% Image size is 990 x 367, HD44780A00 line is in the lower portion
% Provided parameters: [50, 200, 900, 150]
rect = [50, 200, 900, 150];

% Crop the image using custom function
croppedImg = myImcrop(img, rect);

% Save cropped image
imwrite(croppedImg, [outputDir, 'cropped_HD44780A00.png']);

% Create visualization showing the crop region on original
% Save without white borders
figure('Name', 'Task 3: Crop Region Visualization', 'Position', [100, 100, 990, 367], 'Color', 'white');
imshow(img);
hold on;
rectangle('Position', rect, 'EdgeColor', 'r', 'LineWidth', 2);
hold off;

% Save figure without white borders
set(gca, 'Position', [0 0 1 1]);
set(gcf, 'PaperPositionMode', 'auto');
print([outputDir, 'crop_region_highlight.png'], '-dpng', '-r150');

fprintf('\nTask 3 completed successfully!\n');
fprintf('Cropped sub-image saved to: %s\n', outputDir);
fprintf('Crop parameters: [x=%d, y=%d, width=%d, height=%d]\n', rect(1), rect(2), rect(3), rect(4));
