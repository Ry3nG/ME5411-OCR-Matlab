clear; close all; clc;

addpath(genpath('core'));
addpath(genpath('utils'));

imgPath = '../canvas/charact2.bmp';
outputDir = '../output/task1/';

img = imread(imgPath);
if size(img, 3) == 3
    img = rgb2gray(img);
end

figure('Name', 'Task 1: Original Image', 'Position', [100, 100, 800, 600]);
imshow(img);
title('Original Image');
imwrite(img, [outputDir, 'original.png']);

imgHistEq = myHistEq(img);
figure('Name', 'Histogram Equalization', 'Position', [100, 100, 800, 600]);
imshow(imgHistEq);
title('Histogram Equalization');
imwrite(imgHistEq, [outputDir, 'histogram_equalization.png']);

imgStretch = myContrastStretch(img);
figure('Name', 'Contrast Stretching', 'Position', [100, 100, 800, 600]);
imshow(imgStretch);
title('Contrast Stretching');
imwrite(imgStretch, [outputDir, 'contrast_stretching.png']);

gamma_values = [0.5, 1.5, 2.0];
for i = 1:length(gamma_values)
    gamma = gamma_values(i);
    imgGamma = myImadjust(img, [], [], [], [], gamma);
    figure('Name', sprintf('Gamma Correction (gamma=%.1f)', gamma), ...
           'Position', [100, 100, 800, 600]);
    imshow(imgGamma);
    title(sprintf('Gamma Correction (\\gamma = %.1f)', gamma));
    imwrite(imgGamma, [outputDir, sprintf('gamma_%.1f.png', gamma)]);
end

figure('Name', 'Comparison', 'Position', [100, 100, 1600, 1000]);
subplot(2, 3, 1); imshow(img); title('Original');
subplot(2, 3, 2); imshow(imgHistEq); title('Histogram Equalization');
subplot(2, 3, 3); imshow(imgStretch); title('Contrast Stretching');
subplot(2, 3, 4); imshow(myImadjust(img, [], [], [], [], 0.5));
title('Gamma = 0.5');
subplot(2, 3, 5); imshow(myImadjust(img, [], [], [], [], 1.5));
title('Gamma = 1.5');
subplot(2, 3, 6); imshow(myImadjust(img, [], [], [], [], 2.0));
title('Gamma = 2.0');
saveas(gcf, [outputDir, 'comparison.png']);

fprintf('Task 1 completed. Results saved to %s\n', outputDir);
