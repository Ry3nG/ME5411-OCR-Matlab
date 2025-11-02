clear; close all; clc;

addpath(genpath('core'));
addpath(genpath('utils'));

% Input and output directories
binaryImgPath = '../output/task4/binary_otsu.png';
grayscaleImgPath = '../output/task4/grayscale_input.png';
outputDir = '../output/task5/';

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Read the binary image from Task 4
binaryImg = imread(binaryImgPath);
if size(binaryImg, 3) == 3
    binaryImg = rgb2gray(binaryImg);
end

% Read the grayscale image for comparison
grayscaleImg = imread(grayscaleImgPath);
if size(grayscaleImg, 3) == 3
    grayscaleImg = rgb2gray(grayscaleImg);
end

% Convert to logical if needed
if ~islogical(binaryImg)
    binaryImg = binaryImg > 128;
end

fprintf('Starting edge detection...\n');

%% Pre-processing: Clean binary image
fprintf('Cleaning binary image...\n');
% Use morphological opening to remove small noise
se_clean = myStrel('disk', 1);
openedImg = myOpen(uint8(binaryImg) * 255, se_clean);
cleanBinary = logical(openedImg);

%% Method 1: Morphological Boundary Extraction
fprintf('Applying morphological boundary extraction...\n');
% Use erosion-based boundary extraction for clean edges on binary image
se = myStrel('disk', 2);
erodedImg = myErode(uint8(cleanBinary) * 255, se);
morphEdges = cleanBinary & ~logical(erodedImg);  % Boundary = original - eroded

%% Method 2: Sobel Edge Detection
fprintf('Applying Sobel edge detection...\n');
% Apply to cleaned binary image
[sobelEdges, gradMag, gradDir] = mySobel(double(cleanBinary), 0.5);

%% Method 3: Canny Edge Detection
fprintf('Applying Canny edge detection...\n');
% Apply to cleaned binary image for fair comparison
cannyEdges = myCanny(double(cleanBinary), 0.1, 0.3, 1.0);

%% Save individual results
% Save binary input
figure('Name', 'Binary Input', 'Position', [100, 100, 900, 300]);
imshow(binaryImg);
set(gca, 'Position', [0 0 1 1]);
set(gcf, 'PaperPositionMode', 'auto');
print([outputDir, 'binary_input.png'], '-dpng', '-r150');

% Save morphological edges
figure('Name', 'Morphological Boundary', 'Position', [100, 100, 900, 300]);
imshow(morphEdges);
set(gca, 'Position', [0 0 1 1]);
set(gcf, 'PaperPositionMode', 'auto');
print([outputDir, 'morphological_boundary.png'], '-dpng', '-r150');

% Save Sobel edges
figure('Name', 'Sobel Edges', 'Position', [100, 100, 900, 300]);
imshow(sobelEdges);
set(gca, 'Position', [0 0 1 1]);
set(gcf, 'PaperPositionMode', 'auto');
print([outputDir, 'sobel_edges.png'], '-dpng', '-r150');

% Save Canny edges
figure('Name', 'Canny Edges', 'Position', [100, 100, 900, 300]);
imshow(cannyEdges);
set(gca, 'Position', [0 0 1 1]);
set(gcf, 'PaperPositionMode', 'auto');
print([outputDir, 'canny_edges.png'], '-dpng', '-r150');

%% Create comparison figure
figure('Name', 'Task 5: Edge Detection Comparison', 'Position', [100, 100, 2400, 400]);

subplot(1, 4, 1);
imshow(binaryImg);
title('Binary Image (Input)');

subplot(1, 4, 2);
imshow(morphEdges);
title('Morphological Boundary');

subplot(1, 4, 3);
imshow(sobelEdges);
title('Sobel Edges');

subplot(1, 4, 4);
imshow(cannyEdges);
title('Canny Edges');

saveas(gcf, [outputDir, 'comparison.png']);

fprintf('\nTask 5 completed successfully!\n');
fprintf('Edge detection results saved to: %s\n', outputDir);
fprintf('Methods used: Morphological boundary, Sobel, and Canny edge detection\n');
